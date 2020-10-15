# coding=utf8

"""
本文件主要实现文件的预处理：
1. 中文基于jieba分词；
2. 英文基于 Moses分词；
"""

import os
import re
from tqdm import tqdm
from subword_nmt import learn_bpe, apply_bpe
from sacremoses import MosesTokenizer, MosesDetokenizer
from sacremoses import MosesPunctNormalizer
from hb_translate.pre_process import jieba_tokenizer
from hb_translate.utils import str_utils


BPE_TOKENS = 35000
max_words = 200
min_words = 2

space_pat = re.compile(r'\s+')
default_sep = "(。)"
sen_sep_dict = {
    "en": re.compile(r'([.?\n\r]+)'),
    "zh": re.compile(r'([。？\n\r]+)')
}
mpn = MosesPunctNormalizer()


class DataPrepare(object):
    def __init__(self):
        self.en_tok = MosesTokenizer("en")
        self.en_detok = MosesDetokenizer("en")
        self.zh_tok = jieba_tokenizer.Tokenizer()
        self.zh_detok = jieba_tokenizer.DeTokenizer()
        self.tok_dict = {
            "zh": [self.zh_tok, self.zh_detok],
            "en": [self.en_tok, self.en_detok],
        }
        self.lan_list = ["en", "zh"]

    def tokenize(self, line, lang):
        tokenizer = self.tok_dict.get(lang)[0]
        return tokenizer.tokenize(line)


def main(in_dir, out_dir, base_name):
    # import ipdb; ipdb.set_trace()
    preparer = DataPrepare()
    # base_name = "train.lang."
    tmp_dir = os.path.join(out_dir, "tmp")
    _prepare_out_dir(out_dir)
    # 第一步对句子进行clean, 然后对句子进行分词, 并且保存到切好词的地方。
    tok_path_arr = []
    for lang in ["en", "zh"]:
        tok_f_path = _clean_and_tok(in_dir, tmp_dir, base_name, lang, preparer)
        tok_path_arr.append(tok_f_path)

    # 第二步对这两个文件进行过滤；
    filtered_path_arr = _do_filter(tok_path_arr, tmp_dir)

    # 第三步对file 进行split， 分成训练集合测试集
    train_arr, val_arr = _main_split_dir(filtered_path_arr, tmp_dir)

    # 第四步：生成bpe code
    code_arr = _main_create_bpe(out_dir, train_arr)

    # 第五步： 对所有的测试数据以及训练数据进行bpe 编码
    _main_apply_bpe(out_dir, train_arr, code_arr)
    _main_apply_bpe(out_dir, val_arr, code_arr)


def _prepare_out_dir(out_dir):
    os.makedirs(f"{out_dir}/tmp", exist_ok=True)


def _clean_and_tok(in_dir, out_dir, base_name, lang, preparer):
    """
    本函数对in_dir 里面的文件进行拆句子， clean， 以及tokenizer
    """

    in_file = _mk_file_name(in_dir, base_name, lang)
    out_file = _mk_file_name(out_dir, base_name + ".tok", lang)

    with open(in_file) as in_:
        with open(out_file, "w") as out_:
            for line in tqdm(in_, "do tokenizing"):
                line = line.strip()
                sen_list = _clean_sen(line, lang)
                for sen in sen_list:
                    word_list = preparer.tokenize(sen, lang)
                    tok_sen = " ".join(word_list)
                    out_.write(tok_sen + "\n")
    return out_file


def _mk_file_name(dir_name, base_f_name, lang):
    return os.path.join(dir_name, base_f_name + "." + lang)


def _do_filter(f_arr, out_dir):
    """
    本函数实现对平行语料的规则清洗
    """
    visited_lines = set()
    out_arr = _mk_clean_names(f_arr, out_dir)
    with open(f_arr[0]) as _in1:
        with open(f_arr[1]) as _in2:
            with open(out_arr[0], "w") as out_1:
                with open(out_arr[1], "w") as out_2:
                    for line1 in _in1:
                        line1 = line1.strip()
                        line2 = _in2.readline()
                        line2 = line2.strip()
                        if _is_keep(line1, line2, visited_lines):
                            out_1.write(line1 + "\n")
                            out_2.write(line2 + "\n")
    return out_arr


def _mk_clean_names(f_arr, out_dir):
    result = [_mk_clean_name(f_name, out_dir) for f_name in f_arr]
    return result


def _mk_clean_name(f_name, out_dir):
    base_name = os.path.basename(f_name)
    name_arr = base_name.split(".")
    lang = name_arr[-1]
    prefix = ".".join(name_arr[:-1])
    out_name = os.path.join(out_dir, prefix + ".clean" + "." + lang)
    return out_name


def _is_keep(line1, line2, found_lines):
    """
    判断平行语料对是否应该保留
    """
    if not _is_valid_len(line1) or not _is_valid_len(line2):
        return False
    if not _is_len_balance(line1, line2):
        return False
    return _is_not_duplicate(line1, line2, found_lines)


def _is_not_duplicate(line1, line2, found_line):
    md51 = str_utils.str2md5(line1)
    md52 = str_utils.str2md5(line2)
    status = True
    if md51 in found_line or md52 in found_line:
        status = False
    found_line.add(md51)
    found_line.add(md52)
    return status


def _is_valid_len(line):
    len_ = len(line.split())
    if len_ < min_words or len_ > max_words:
        return False
    return True


def _is_len_balance(line1, line2):
    len1 = len(line1.split())
    len2 = len(line2.split())
    if len1 > 1.5 * len2:
        return False
    if len2 > 1.5 * len1:
        return False
    return True


def _clean_sen(sen, lang):
    # 如果句子是英文， 对英文进行normalize
    if lang == "en":
        sen = mpn.normalize(sen)
    # 第一步对句子进行句号拆分, 这一步暂时还有问题，因此不能做了
    # sen_sep = sen_sep_dict.get(lang, default_sep)
    # sen_list = re.split(sen_sep, sen) + [""]
    # sen_list = ["".join(elem) for elem in zip(sen_list[0::2], sen_list[1::2])]
    sen_list = [sen]
    # 第二步对句子进行修整和空格替换
    sen_list = [(re.sub(space_pat, " ", sub_sen)).strip() for sub_sen in sen_list]
    return sen_list


def _main_split_dir(f_arr, out_dir):
    """
    对语料进行split, 生成平行语料的train 和 valid 文件
    """
    train_res, val_res = [], []
    for f_path in f_arr:
        suffix = f_path.split(".")[-1]
        train_path, val_path = _split_file(f_path, suffix, out_dir)
        train_res.append(train_path)
        val_res.append(val_path)
    return train_res, val_res


def _split_file(in_file, out_suffix, out_dir):
    train_file = os.path.join(out_dir, "train." + out_suffix)
    val_file = os.path.join(out_dir, "val." + out_suffix)
    with open(in_file) as in_:
        with open(train_file, "w") as out_tra:
            with open(val_file, "w") as out_val:
                for i, line in enumerate(in_):
                    if i % 23 == 0:
                        out_val.write(line)
                    else:
                        out_tra.write(line)
    return train_file, val_file


def _main_create_bpe(out_dir, f_arr):
    """
    :param 构建
    """
    res = []
    for f_path in f_arr:
        lang = f_path.split(".")[-1]
        code_path = os.path.join(out_dir, "code" + "." + lang)
        _cons_bpe(f_path, code_path)
        res.append(code_path)
    return res


def _cons_bpe(in_path, code_path):
    with open(in_path) as in_:
        with open(code_path, "w") as out_:
            learn_bpe.learn_bpe(in_, out_, BPE_TOKENS)


def _main_apply_bpe(out_dir, f_arr, code_arr):
    code_path = code_arr[0]
    for f_path in f_arr:
        lang = f_path.split(".")[-1]
        if not code_path.endswith(lang):
            code_path = code_arr[1]
        out_path = os.path.join(out_dir, os.path.basename(f_path))
        _apply_bpe(out_path, f_path, code_path)


def _apply_bpe(out_path, f_path, code_path):
    with open(code_path) as f_code:
        bpe = apply_bpe.BPE(f_code)
        with open(out_path, "w") as out_:
            with open(f_path) as in_:
                for line in tqdm(in_, "applying bpe"):
                    out_.write(bpe.process_line(line))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--fbase', type=str)
    args = parser.parse_args()
    main(args.input, args.output, args.fbase)
