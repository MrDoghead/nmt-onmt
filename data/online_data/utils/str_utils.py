# coding=utf8

import re
import hashlib


upper_pat = re.compile(r'^[A-Z.]+$')

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
