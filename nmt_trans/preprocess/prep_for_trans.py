# coding=utf8

"""
本文件属于处理平行语料的第一步:
对要替换的词进行替换
"""

import os
from nmt_trans.preprocess import add_tag
from nmt_trans.utils import file_helper
from nmt_trans.preprocess import tok_and_clean


class RawPreprocessor(object):
    def __init__(self, conf):
        self.conf = conf
        self.org_corpus_dir = self.conf.raw_org_data
        self.org_tag_dir = self.conf.raw_tag_data
        self.prep_dir = self.conf.preded_dir
        self.prep_base_name = self.conf.preped_base_name
        dict_path = file_helper.get_online_data("caijing_clean.csv")
        self.tag_adder = add_tag.PrecessTag(dict_path)
        self.mk_dirs()

    def mk_dirs(self):
        dir_arr = [
            self.org_tag_dir,
            self.prep_dir
        ]
        for dir_ in dir_arr:
            os.makedirs(dir_, exist_ok=True)

    def run(self):
        self.tag_adder.process_en(self.org_corpus_dir, self.org_tag_dir, self.prep_base_name)
        tok_and_clean.main(self.org_tag_dir, self.prep_dir, self.prep_base_name)
