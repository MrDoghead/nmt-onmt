# coding=utf8

import json
import os


class En2zhDict(object):
    def __init__(self, words_dir):
        self.en2zh = self._load_words(words_dir)

    def translate(self, word):
        word = word.strip().lower()
        return self.en2zh.get(word, None)

    def _load_words(self, in_dir):
        res = dict()
        f_list = os.listdir(in_dir)
        for f in f_list:
            if f.endswith(".josn"):
                f_path = os.path.join(in_dir, f)
                self._load_json(f_path, res)
        return res

    def _load_json(self, f_path, res):
        with open(f_path) as in_:
            for line in in_:
                line = line.strip()
                if len(line) > 1:
                   obj = json.loads(line)
                   word = word