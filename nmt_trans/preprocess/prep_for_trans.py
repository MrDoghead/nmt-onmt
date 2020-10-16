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
        self.org_corpus_dir = file_helper.get_real_path(self.conf.raw_org_data)
        self.org_tag_dir = file_helper.get_abs_path(self.conf.raw_tag_data)
        self.prep_dir = file_helper.get_abs_path(self.conf.preped_dir)
        self.prep_base_name = self.conf.preped_base_name
        dict_path = file_helper.get_online_data("caijing_clean.csv")
        self.tag_adder = add_tag.PrecessTag(dict_path)
        self.mk_dirs()
        self.bpe_path = file_helper.get_online_data(self.conf.bpe_code_path)

    def mk_dirs(self):
        dir_arr = [
            self.org_tag_dir,
            self.prep_dir
        ]
        for dir_ in dir_arr:
            os.makedirs(dir_, exist_ok=True)

    def run(self):
        self.tag_adder.process_en(self.org_corpus_dir, self.org_tag_dir, self.prep_base_name)
        print("finished adding data")
        tok_and_clean.main(self.org_tag_dir, self.prep_dir, self.prep_base_name)
        # 最后将bpe 放到online_data 为便于预测
        file_helper.cp_f(os.path.join(self.prep_dir, "code.en"), self.bpe_path)
        file_helper.cp_f(os.path.join(self.prep_dir, "code.zh"), self.bpe_path)


if __name__ == "__main__":
    from nmt_trans.utils import conf_parser
    import sys
    conf_f = sys.argv[1]
    t_conf = conf_parser.parse_conf(conf_f)
    t_obj = RawPreprocessor(t_conf)
    t_obj.run()
