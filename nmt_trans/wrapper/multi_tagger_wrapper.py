# coding=utf8

"""
本文件主要用来进行数据的预处理和后处理，可以理解为预处理的时候包了包装， 翻译过后解了包装
这个文件主要是解决同时出现多个需要替换的实体时，
"""

import re
from collections import namedtuple, OrderedDict

WrapperInfo = namedtuple('WrapperInfo', 'map_arr')


class MultiTagWrapper(object):
    multi_pat = r'(?:(?:｟\d+｠[，、,\s]+)+｟\d+｠)|(?:｟\d+｠)+'
    single_pat = r'｟\d+｠'

    @classmethod
    def pre_process(cls, text_arr):
        res = []
        map_arr = []
        for v_no, text in enumerate(text_arr):
            n_text, repl_map = cls.preprocess_impl(text)
            if n_text is not None:
                res.append(n_text)
                map_arr.append(repl_map)
        wrap_info = WrapperInfo(map_arr)
        return res, wrap_info

    @classmethod
    def preprocess_impl(cls, text):
        if text is None:
            return None, None
        found_arr = re.findall(cls.multi_pat, text)
        idx = 0
        _map = dict()
        for f in found_arr:
            if f not in _map:
                _map[f] = idx
                idx += 1
        res_map = dict()
        for key, val in _map.items():
            repl_str = f"｟{val}｠"
            text = text.replace(key, repl_str)
            res_map[repl_str] = key
        return text, res_map

    @classmethod
    def post_process(cls, text_arr, wrap_info):
        map_arr = wrap_info.map_arr
        res = []
        for i, text in enumerate(text_arr):
            text = cls._multi_rep(text, map_arr[i])
            res.append(text)
        return res

    @classmethod
    def _multi_rep(cls, text, rep_map):
        res_arr = []
        m_arr = re.finditer(cls.single_pat, text)
        begin = 0
        for m in m_arr:
            s = m.start()
            e = m.end()
            key = text[s:e]
            if key in rep_map:
                val = rep_map[key]
                val = val.replace("、", ", ").replace("，", ", ")
                res_arr.append(text[begin:s])
                res_arr.append(val)
                begin = e
        if begin < len(text):
            res_arr.append(text[begin:])
        return "".join(res_arr)


if __name__ == "__main__":
    test_arr = ["这家｟0｠芯片开发商从中国国有集成电路基金,｟0｠、｟1｠,｟2｠,｟3｠,｟4｠,｟5｠。"]
    test_arr = ["这家｟0｠、｟1｠芯片开发商从中国国有集成电路基金,｟1｠,｟2｠,｟3｠,｟4｠,｟5｠。"]
    t_res, t_info = MultiTagWrapper.pre_process(test_arr)
    print(t_res)
    print(t_info)
    t_res = MultiTagWrapper.post_process(t_res, t_info)
    print(t_res)

