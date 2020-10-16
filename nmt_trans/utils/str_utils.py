# coding=utf8

import re
import hashlib
from nmt_trans.tokenizer.basic_tokenizer import BaseTokenizer


upper_pat = re.compile(r'^[A-Z.]+$')
tokenizer = BaseTokenizer()
sub_en_sen_pat = re.compile(r'([,，\n.]+)')


def str2md5(str_):
    str_ = str_.encode()
    hash_obj = hashlib.md5(str_)
    return hash_obj.hexdigest()


def preprocess_eng(str_):
    """
    本文将英文的句子处了全部是是大写的外，都改写为小写；
    """
    tmp_arr = []
    words = str_.split()
    for word in words:
        if is_all_upper(word):
            tmp_arr.append(word)
        else:
            tmp_arr.append(word.lower())
    return " ".join(tmp_arr)


def is_all_upper(word):
    if re.match(upper_pat, word):
        return True
    return False


def remove_repeat(text):
    char_arr = tokenizer.tokenize(text)
    pre_arr = _get_pre_arr(char_arr)
    del_idxs = _get_del_ids(pre_arr)
    res_text = []
    for i, ch in enumerate(char_arr):
        if i not in del_idxs:
            res_text.append(ch)
    return "".join(res_text)


def _get_pre_arr(char_arr):
    res = []
    pre_dict = dict()
    for i, ch in enumerate(char_arr):
        pre_pos = pre_dict.get(ch, -1)
        res.append(pre_pos)
        pre_dict[ch] = i
    return res


def _get_del_ids(pre_arr):
    res = set()
    tmp_del = []
    tmp_pre = -1
    for i, idx in enumerate(pre_arr):
        if idx >= 0:
            if tmp_pre == -1:
                tmp_pre = i
                tmp_del.append(idx)
            else:
                # 第一步判断是否是同步
                if not _is_consective(idx, tmp_del):
                    # 不同步
                    tmp_del = []
                    tmp_pre = i
                tmp_del.append(idx)

            if idx == tmp_pre - 1:
                res.update(tmp_del)
                tmp_pre = -1
                tmp_del = []
        else:
            tmp_del = []
            tmp_pre = -1
    return res


def _is_consective(elem, _arr):
    if len(_arr) < 1:
        return True
    if _arr[-1] + 1 == elem:
        return True
    return False


def split_sub_sen(sen):
    if sen is None:
        return None
    sub_arr = re.split(sub_en_sen_pat, sen)
    if len(sub_arr) % 2 != 0:
        sub_arr.append("")
    a1 = sub_arr[::2]
    a2 = sub_arr[1::2]
    x = zip(a1, a2)
    res = ["".join(elem) for elem in x]
    return res


if __name__ == "__main__":
    test_str = "So we’ve moved from beige to bold. But why? For Sarokin, these strong colours are an outward reflection of the strong female spirit that is part of the current cultural zeitgeist. “It’s a reflection of the times,” she says. “Women are more passionate and individual, and that is mirrored in the strength and boldness we see in colour palettes in fashion.”"
    cvt_str = preprocess_eng(test_str)
    print(cvt_str)
    print(len(cvt_str.split()))
    test_str = "我要我要我要看书， 我要我要写字字2020, good food food"
    cl_str = remove_repeat(test_str)
    print(cl_str)
