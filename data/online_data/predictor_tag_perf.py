# coding=utf8

import re
import fileinput
import os
import torch
from collections import namedtuple
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from hb_translate.pre_process import custom_tag

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
        self.tokenizer = encoders.build_tokenizer(args)
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

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
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

    def predict(self, input_sens):
        # import ipdb; ipdb.set_trace()
        inputs, en_words_arr = self.pre_process(input_sens)
        results = []
        batches = make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn)
        cnt = 0
        for batch in batches: 
            cnt += 1
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
           

        print("batch_num is: ", cnt)
        f_result = []
        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                # print('S-{}\t{}'.format(id, src_str))

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
            ff_result.append(self._replace_use_dict(en_words_arr[i], out_sen))

        return ff_result

    def pre_process(self, sens):
        res_sen = []
        res_dict = []
        for sen in sens:
            out_sen, out_map = self.encode_en(sen)
            res_sen.append(out_sen)
            res_dict.append(out_map)
        return res_sen, res_dict

    def encode_en(self, sen):
        en_tags = self.tag_helper.search_en(sen)
        ent_map = {} # 实体名字到index 的对应
        if len(en_tags) < 1:
            return sen, ent_map
        en_words = [sen[s:e+1] for s, e in en_tags]
        ent_map = self.words2dict(en_words)
        f_en_pos_2_tag = {}
        for i, en_word in enumerate(en_words):
            s, e = en_tags[i]
            f_en_pos_2_tag[(s, e+1)] = str(ent_map[en_word])
        new_enline = self._replace_line(sen, f_en_pos_2_tag)
        return new_enline, ent_map

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

    def _replace_line(self, origin_line, pos_word_list):
        line_infos = self._split_lines(origin_line, pos_word_list.keys())
        sub_lines = []
        for key, sub_line in line_infos:
            if key in pos_word_list:
                sub_line = f"<ent> {pos_word_list[key]} <\ent>"
            sub_lines.append(sub_line)
        return "".join(sub_lines)

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


if __name__ == "__main__":
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    cus_word_path = "online_data/caijing_clean.csv"
    predictor = Predictor(args, cus_word_path)
    # test_str = [
    #     "Hello, world",
    #     "UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier. More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year. U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year."
    # ]

    text_list = ["UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier.",  "More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year." ,"U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year.","China reported 11 new confirmed coronavirus cases in the mainland as of end-Sunday (May 24), up from three a day earlier, the National Health Commission reported. ", "This year the holiday week is when the US death toll from COVID-19 is expected to exceed 100,000. ", "For Tesla Motors, cutting edge technology is everything. Musk, an engineer himself, focused right from the beginning on an ultimate reliable technology product. ", "The Indian automobile market is at its crossroads. The demanding environment norms are making the OEMs invest continuously in their IC engine vehicles.", "In a high price-sensitive market, it is eating into their profitability. ", ""]
    import cProfile
    import pstats
    import sys
    prof = cProfile.Profile()
    prof.enable()
    trans_result = predictor.predict(text_list)
    prof.disable()
    p = pstats.Stats(prof, stream=sys.stdout)
    p.sort_stats('cumulative').print_stats(100)
    # for res in trans_result:
    #    print(res)
