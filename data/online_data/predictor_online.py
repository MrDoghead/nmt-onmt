# coding=utf8

import re
import fileinput
import os
import torch
from collections import namedtuple
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from hb_translate.pre_process import custom_tag
from hb_translate.pre_process import mose_tokenizer
from hb_translate.pre_process import jieba_tokenizer  
from hb_translate.utils.trans_dict_manager import EnZhMapper
from hb_translate.utils.str_utils import  split_sub_sen

no_space_ch_pat = re.compile(r'[\u4e00-\u9fa5<>\\()（）]+')
ent_pat = re.compile(r'<ent>(.+?)<\\ent>')
uneed_pat = re.compile(r'\([0-9a-zA-z\s]+\)|>\s+')
noise_pat = re.compile(r'<ent>[\d\s]+<\\ent>|>[\d\s]+<\\ent>|<ent>[\d\s]+<|<ent>|<\\ent>|<|>')

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


class Predictor(object):
    def __init__(self, args, cus_dict):
        utils.import_user_module(args)
        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task, e.g., translation
        task = tasks.setup_task(args)

        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            args.path.split(os.pathsep),
            arg_overrides=eval(args.model_overrides),
            task=task,
        )

        # Set dictionaries
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()

        # Initialize generator
        self.generator = task.build_generator(self.models, args)

        # Handle tokenization and BPE
        # self.tokenizer = encoders.build_tokenizer(args)
        self.tokenizer = mose_tokenizer.MTokenizer("en")
        self.jb_tokenizer = jieba_tokenizer.Tokenizer()
        self.bpe = encoders.build_bpe(args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in self.models]
        )
        self.use_cuda = use_cuda
        self.args = args
        self.task = task
        self.tag_helper = custom_tag.TagHelper(cus_dict)
        self.en_zh_mapper = EnZhMapper()

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.tokenize(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        word_list = x.split() 
        _need_sp = False
        result = []
        for word in word_list:
            if not self._is_no_sapce_ch(word):
                if _need_sp:
                    result.append(" ")
                _need_sp = True
            else:
               _need_sp = False 
            result.append(word)
        return "".join(result)

    def _is_no_sapce_ch(self, x):
        return re.match(no_space_ch_pat, x)

    def _predict_impl(self, input_sens):
        emp_idxes = set()
        map_idxes = dict()
        for i, sen in enumerate(input_sens):
            if len(sen.strip()) == 0:
                emp_idxes.add(i)
            else:
                zh_sen = self.en_zh_mapper.get_trans(sen.strip())
                if zh_sen is not None:
                    map_idxes[i] = zh_sen

        inputs, en_words_arr = self.pre_process(input_sens)
        results = []
        for batch in make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((id, src_tokens_i, hypos))

        f_result = []
        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                detok_hypo_str = self.decode_fn(hypo_str)
                f_result.append(detok_hypo_str)
        ff_result = []
        for i, out_sen in enumerate(f_result):
            if i in emp_idxes:
                ff_result.append(" ")
            elif i in map_idxes:
                ff_result.append(map_idxes[i])
            else:
                ff_result.append(self.tag_helper.decode_zh(out_sen, en_words_arr[i]))

        ff_result = self.final_proc(ff_result)
        return ff_result

    def pre_process(self, sens):
        res_sen = []
        res_dict = []
        for sen in sens:
            out_sen, out_map = self.tag_helper.encode_en(sen)
            res_sen.append(out_sen)
            res_dict.append(out_map)
        return res_sen, res_dict

    def _split_lines(self, line, pos_list):
        sorted_list = sorted(pos_list, key=lambda x: x[0])
        flat_list = [item for sublist in sorted_list for item in sublist]
        begin = 0
        line_infos = []
        for elem in flat_list:
            key = (begin, elem)
            sub_line = line[begin:elem]
            line_infos.append([key, sub_line])
            begin = elem
        if begin < len(line):
            sub_line = line[begin:]
            key = (begin, len(line))
            line_infos.append([key, sub_line])
        return line_infos

    def words2dict(self, word_list):
        res_dict = {}
        cnt = 1
        for word in word_list:
            if word not in res_dict:
                res_dict[word] = cnt
                cnt += 1
        return res_dict

    def _replace_use_dict(self, en_map, out_sen):
        if len(en_map) < 1:
            return out_sen
        id2en = dict([(str(v), k) for k, v in en_map.items()])
        match_arr = re.finditer(ent_pat, out_sen)
        new_sen = out_sen
        for match in match_arr:
            match_str = match.group(0)
            en_id = match.group(1).strip()
            en_str = id2en.get(en_id)
            if en_str is not None:
                zh_str = self.tag_helper.get_zh(en_str)
                if zh_str is not None:
                    new_sen = new_sen.replace(match_str, zh_str)
        new_sen = re.sub("<ent>[\d\s]+<\\\\ent>", "", new_sen)
        new_sen = new_sen.replace("<ent>", "")
        new_sen = new_sen.replace("<\\ent>", "")
        new_sen = re.sub(uneed_pat, "", new_sen)
        return new_sen

    def update_tag_words(self, mod_dict):
        self.tag_helper.update(mod_dict)

    def update_tag_batchly(self):
        self.tag_helper.update_batchly()

    def final_proc(self, sens):
        res = []
        for sen in sens:
            res.append(self._repl_with_dict(sen))
        return res

    def _repl_with_dict(self, sen):
        words = self.jb_tokenizer.tokenize(sen)
        res = []
        for word in words:
            rep_word = self.en_zh_mapper.get_trans(word)
            if rep_word is None:
                res.append(word)
            else:
                res.append(rep_word)
        new_sen = "".join(res)
        new_sen = self.decode_fn(new_sen) 
        return new_sen

    def _split_sub_sen(self, sen):
        """
        用来切分句子， 当句子长度超过150个单词的时候，
        """
        thres = 100
        if len(sen.strip().split()) < thres:
            return [sen]
        res = split_sub_sen(sen)
        return res

    def split_sub_sens(self, sens):
        """
        将句子全部切分出来， 主要用来切分容易过长的句子；
        """
        begin = 0
        sen_no_map = dict()
        sub_sens = []
        for i, sen in enumerate(sens):
            cur_sub_sens = self._split_sub_sen(sen)
            if cur_sub_sens is not None:
                sub_sens.extend(cur_sub_sens)
            for j in range(len(cur_sub_sens)):
                sen_no_map[begin+j] = i
                begin += len(cur_sub_sens)
        return sub_sens, sen_no_map

    def merge_sub_sens(self, sub_sens, sen_no_map):
        mid_sens = [""] * len(sen_no_map)
        for i, sub_sen in enumerate(sub_sens):
            j = sen_no_map[i]
            mid_sens[j] = mid_sens[j] + sub_sen
        return mid_sens

    def predict(self, input_sens):
        sub_sens, sen_no_map = self.split_sub_sens(input_sens)
        trans_res = self._predict_impl(sub_sens)
        res = self.merge_sub_sens(trans_res, sen_no_map)
        return res


if __name__ == "__main__":
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    # cus_word_path = "data/caijin.csv"
    cus_word_path = "online_data/caijing_clean.csv"
    predictor = Predictor(args, cus_word_path)
    test_str = [
        "Hello, world",
        "UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier. More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year. U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year.",
       "Founded 2019 by CEO Ajay Gupta and Candice Gupta, Stirista is a data-driven digital marketing solutions provider for brands to increase conversions and customer retention. Customers include Fortune 500 and mid-market brands including Great Clips, Oracle and Verizon, among others" 
    ]
    # import ipdb; ipdb.set_trace()
    trans_result = predictor.predict(test_str)
    for res in trans_result:
        print(res)
