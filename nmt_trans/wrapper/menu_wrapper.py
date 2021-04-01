# coding=utf8

"""
本文件主要用来进行数据的预处理和后处理，可以理解为预处理的时候包了包装， 翻译过后解了包装
这个文件主要是解决pdf 翻译中遇到的目录处理问题
"""

import re
from collections import namedtuple, OrderedDict

WrapperInfo = namedtuple('WrapperInfo', 'sen_no_map menu_info_arr')


class MenuWrapper(object):
    menu_pat = r'(' \
                '[-.]{3,}\s*\d*' \
                '|\s+\d+$' \
                '|[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)+' \
                '|\(?（?(?:https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]\)?）?' \
                ')'

    @classmethod
    def pre_process(cls, text_arr):
        sub_sen_no = 0
        res = []
        menu_arr = []
        no_map = {}
        for v_no, text in enumerate(text_arr):
            sub_sen_arr, sep_arr = split_by_pat(text, cls.menu_pat)
            for sub_sen in sub_sen_arr:
                if len(sub_sen) > 0:
                    no_map[sub_sen_no] = v_no
                    sub_sen_no += 1
                    res.append(sub_sen)
            menu_arr.append(sep_arr)
        wrap_info = WrapperInfo(no_map, menu_arr)
        return res, wrap_info

    @classmethod
    def post_process(cls, text_arr, wrap_info):
        no_map = wrap_info.sen_no_map
        menu_info = wrap_info.menu_info_arr
        org_no_sen_dict = OrderedDict()
        for i, text in enumerate(text_arr):
            if i in no_map:
                org_no = no_map[i]
                if org_no not in org_no_sen_dict:
                    org_no_sen_dict[org_no] = []
                org_no_sen_dict[org_no].append(text)
        res_arr = []
        for org_no, text_list in org_no_sen_dict.items():
            if org_no < len(menu_info):
                suffix_arr = menu_info[org_no]
            else:
                suffix_arr = []
            res_text = cls._merge_text_list(text_list, suffix_arr)
            res_arr.append(res_text)
        return res_arr
    
    @classmethod
    def post_process_with_states(cls, text_arr, states, wrap_info):
        '''
        后处理合并句子的同时合并states(句子错误状态)
        '''
        no_map = wrap_info.sen_no_map
        menu_info = wrap_info.menu_info_arr
        org_no_sen_dict = OrderedDict()
        org_no_state_dict = OrderedDict()
        for i, text in enumerate(text_arr):
            state = states[i]
            if i in no_map:
                org_no = no_map[i]
                if org_no not in org_no_sen_dict:
                    org_no_sen_dict[org_no] = []
                    org_no_state_dict[org_no] = []
                org_no_sen_dict[org_no].append(text)
                org_no_state_dict[org_no].append(state)
        res_arr = []
        res_states = []
        for org_no, text_list in org_no_sen_dict.items():
            state_list = org_no_state_dict[org_no]
            if org_no < len(menu_info):
                suffix_arr = menu_info[org_no]
            else:
                suffix_arr = []
            res_text = cls._merge_text_list(text_list, suffix_arr)
            res_state = cls._merge_state_list(state_list)
            res_arr.append(res_text)
            res_states.append(res_state)
        return res_arr, res_states
    
    
    @classmethod
    def _merge_text_list(cls, text_list, suffix_arr):
        if len(suffix_arr) < 1:
            return "".join(text_list)
        if suffix_arr is None:
            return "".join(text_list)
        res = []
        for i, suffix in enumerate(suffix_arr):
            if i < len(text_list):
                res.append(text_list[i])
                res.append(suffix)
        if len(suffix_arr) < len(text_list):
            res.extend(text_list[len(suffix_arr):])
        return "".join(res)
    
    @classmethod
    def _merge_state_list(cls, state_list):
        return sum(state_list) == len(state_list)
        

def split_by_pat(text, pat):
    """
    :param text: 输入的句子
    :param pat: 用来切分句子的正则表达式子
    :return:
    """
    res_arr = re.split(pat, text)
    sen_arr = res_arr[::2]
    sep_arr = res_arr[1::2]
    return sen_arr, sep_arr


if __name__ == "__main__":
    test_arr = ["1.2     中国的大数据红利在机器学习时代将充分迸发 .............................................  11 哈哈哈-----12",
                "1.2 我们的国家",
                "图 126  复购客户收入保持逐年增长        71",
                "根据中南集团•中南建设官网（http://www.zhongnangroup.cn）披露：中南集团“2018年目标",
                "请发送到邮箱1234567@fool.bar, 我们会尽快回复"]
    res_arr, menu_info = MenuWrapper.pre_process(test_arr)
    print(res_arr)
    print(menu_info)
    new_res_arr = MenuWrapper.post_process(res_arr, menu_info)
    print(new_res_arr)
    print("orgin input is: ")
    print(test_arr)
