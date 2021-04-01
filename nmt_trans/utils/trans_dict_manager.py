# coding=utf8

import os
import json
from nmt_trans.utils import file_helper


class EnZhMapper(object):
    def __init__(self):
        dir_path = file_helper.get_online_data("word_map")
        self.word_map = self._load_word_map(dir_path)

    def _load_word_map(self, _dir):
        res = dict()
        f_list = os.listdir(_dir)
        for f in f_list:
            if f.endswith(".json"):
                f_path = os.path.join(_dir, f)
                tmp_dict = self._load_word_from_f(f_path)
                res.update(tmp_dict)
        return res

    def _load_word_from_f(self, f_path):
        res = dict()
        with open(f_path) as in_:
            for line in in_:
                line = line.strip()
                if len(line) > 2:
                    obj_ = json.loads(line)
                    word = obj_["headWord"].strip().lower()
                    raw_trans = obj_["content"]["word"]["content"]["trans"][0]["tranCn"]
                    raw_trans = raw_trans.split("；")[0].strip()
                    raw_trans = raw_trans.split("（")[0].strip()
                    raw_trans = raw_trans.split("(")[0].strip()
                    if len(word) > 0 and len(raw_trans) > 0:
                        res[word] = raw_trans
        return res

    def get_trans(self, sen):
        sen = sen.strip().lower()
        return self.word_map.get(sen, None)


class ZhEnMapper(object):
    def __init__(self):
        dir_path = file_helper.get_online_data("word_map")
        self.word_map = self._load_word_map(dir_path)

    def _load_word_map(self, _dir):
        res = dict()
        f_list = os.listdir(_dir)
        for f in f_list:
            if f.endswith(".json"):
                f_path = os.path.join(_dir, f)
                tmp_dict = self._load_word_from_f(f_path)
                res.update(tmp_dict)
        return res

    def _load_word_from_f(self, f_path):
        res = dict()
        with open(f_path) as in_:
            for line in in_:
                line = line.strip()
                if len(line) > 2:
                    obj_ = json.loads(line)
                    word = obj_["headWord"].strip().lower()
                    raw_trans = obj_["content"]["word"]["content"]["trans"][0]["tranCn"]
                    raw_trans = raw_trans.split("（")[0].strip()
                    raw_trans = raw_trans.split("(")[0].strip()
                    raw_trans = raw_trans.split("；")[0].strip()
                    if len(word) > 0 and len(raw_trans) > 0:
                        res[raw_trans] = word
        return res

    def get_trans(self, sen):
        sen = sen.strip().lower()
        return self.word_map.get(sen, None)


if __name__ == "__main__":
    t_mapper = EnZhMapper()
    print(len(t_mapper.word_map))
    print(t_mapper.get_trans("Hello"))
    t_mapper2 = ZhEnMapper()
    print(t_mapper2.get_trans("中国"))
