# coding=utf8
"""
本文件主要是用来保留那些常用的转换
"""


def list2dict(word_list, start=0):
    res_dict = {}
    cnt = start
    for word in word_list:
        if word not in res_dict:
            res_dict[word] = cnt
            cnt += 1
    return res_dict


def reverse_key_val(dict_):
    if dict_ is None:
        return None
    return dict([(v, k) for k, v in dict_.items()])
