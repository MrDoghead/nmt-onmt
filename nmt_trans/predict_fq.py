# coding=utf8

"""
本文件主要实现基于fairseq来翻译英译中的方法
"""
import sys
import os
import torch
import ast
import logging
import re
from collections import namedtuple
from argparse import Namespace
import numpy as np

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import options
from subword_nmt import apply_bpe

from nmt_trans.utils import file_helper, conf_parser
from nmt_trans.tokenizer import jieba_tokenizer
from nmt_trans.tokenizer import mose_tokenizer
from nmt_trans.utils.trans_dict_manager import EnZhMapper
from nmt_trans.tags import custom_tag
from nmt_trans.utils.str_utils import split_sub_sen

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

no_space_ch_pat = re.compile(r'[\u4e00-\u9fa5<>\\()（）]+')
ent_pat = re.compile(r'<ent>(.+?)<\\ent>')
uneed_pat = re.compile(r'\([0-9a-zA-z\s]+\)|>\s+')
noise_pat = re.compile(r'<ent>[\d\s]+<\\ent>|>[\d\s]+<\\ent>|<ent>[\d\s]+<|<ent>|<\\ent>|<|>')


class Predictor(object):
    def __init__(self, conf, dict_path):
        self.conf = conf
        fq_conf = conf.fairseq_conf
        self.input_data = file_helper.get_abs_path(fq_conf.preprocess.destdir)
        # 依据conf找到模型文件地址
        self.model_path = os.path.join(file_helper.get_abs_path(fq_conf.train.save_dir),
                                       fq_conf.infer.path)
        self.args = self._init_args()
        if isinstance(self.args, Namespace):
            self.args = convert_namespace_to_omegaconf(self.args)
        self._init_model(self.args)
        self.bpe = self._load_bpe()
        self.bpe_symbol = "@@ "
        self.tokenizer = mose_tokenizer.MTokenizer("en")
        self.jb_tokenizer = jieba_tokenizer.Tokenizer()
        self.tag_helper = custom_tag.TagHelper(dict_path)
        self.en_zh_mapper = EnZhMapper()

    def _init_args(self):
        parser = options.get_interactive_generation_parser()
        args = self.get_args(parser)
        return args

    def get_args(self, parser):
        args_dict = {
            "path": self.model_path,
        }
        conf_dict = conf_parser.conf2dict(self.conf.fairseq_conf.infer)
        conf_dict.update(args_dict)
        args_arr = conf_parser.dict2args(conf_dict)
        f_args_arr = [self.input_data]
        f_args_arr.extend(args_arr)
        args = options.parse_args_and_arch(parser, input_args=f_args_arr)
        return args

    def _init_model(self, cfg):
        utils.import_user_module(cfg.common)
        if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
            cfg.dataset.batch_size = 16 

        logger.info(cfg)
        # Fix seed for stochastic decoding
        if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
            np.random.seed(cfg.common.seed)
            utils.set_torch_seed(cfg.common.seed)

        self.use_cuda = torch.cuda.is_available() and not cfg.common.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(cfg.task)

        # Load ensemble
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=self.task,
        )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if self.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, cfg.generation)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(cfg.generation.replace_unk)
        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )
        logger.info("NOTE: hypothesis and token scores are output in base 2")
        logger.info("Type the input sentence and press return:")

    def _load_bpe(self):
        bpe_path = file_helper.get_real_path(self.conf.bpe_code_path)
        bpe_path = os.path.join(bpe_path, "code.en")
        with open(bpe_path) as in_:
            bpe = apply_bpe.BPE(in_)
            return bpe

    def predict(self, input_sens):
        sub_sens, sen_no_map = self.split_sub_sens(input_sens)
        trans_res = self._predict_impl(sub_sens)
        res = self.merge_sub_sens(trans_res, sen_no_map)
        return res

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
        batches = make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn)
        results = []
        for batch in batches:
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((id, src_tokens_i, hypos,))

        f_results = []
        for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.common_eval.post_process)
                print("S-{}\t{}".format(id_, src_str))

            # Process top predictions
            for hypo in hypos[: min(len(hypos), self.args.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.common_eval.post_process,
                )
                detok_hypo_str = self.decode_fn(hypo_str)
                f_results.append(detok_hypo_str)

        ff_result = []
        for i, out_sen in enumerate(f_results):
            if i in emp_idxes:
                ff_result.append(" ")
            elif i in map_idxes:
                ff_result.append(map_idxes[i])
            else:
                ff_result.append(self.tag_helper.decode_zh(out_sen, en_words_arr[i])[0])

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

    def update_tag_words(self, mod_dict):
        self.tag_helper.update(mod_dict)

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.tokenize(x)
        if self.bpe is not None:
            x = self.bpe.process_line(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self._decode_bpe(x)
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

    def _decode_bpe(self, x: str) -> str:
        return (x + ' ').replace(self.bpe_symbol, '').rstrip()

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
            if i in sen_no_map:
                j = sen_no_map[i]
                mid_sens[j] = mid_sens[j] + sub_sen
        return mid_sens


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)
        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def test(conf_path, dict_path):
    conf = conf_parser.parse_conf(conf_path)
    t_predictor = Predictor(conf, dict_path)
    test_str = [
        "Hello, world",
        "UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier. More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year. U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year.",
        "Founded 2019 by CEO Ajay Gupta and Candice Gupta, Stirista is a data-driven digital marketing solutions provider for brands to increase conversions and customer retention. Customers include Fortune 500 and mid-market brands including Great Clips, Oracle and Verizon, among others"
    ]
    res = t_predictor.predict(test_str)
    print(res)


if __name__ == "__main__":
    import sys
    t_conf_path = sys.argv[1]
    t_dict_path = sys.argv[2]
    test(t_conf_path, t_dict_path)


