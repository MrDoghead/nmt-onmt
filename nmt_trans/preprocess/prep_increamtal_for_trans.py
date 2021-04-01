# coding=utf8

"""
本文件属于处理平行语料的第一步:
对要替换的词进行替换
"""

import os
from nmt_trans.utils import file_helper
from nmt_trans.preprocess import tok_and_clean_increamental as processor


class RawPreprocessor(object):
    def __init__(self, conf):
        self.conf = conf
        self.org_corpus_dir = file_helper.get_real_path(self.conf.raw_org_data)
        self.prep_dir = file_helper.get_abs_path(self.conf.preped_dir)
        self.prep_base_name = self.conf.preped_base_name
        self.mk_dirs()
        self.bpe_path = file_helper.get_abs_path(self.conf.bpe_code_path)

    def mk_dirs(self):
        dir_arr = [
            self.prep_dir
        ]
        for dir_ in dir_arr:
            os.makedirs(dir_, exist_ok=True)

    def run(self):
        bpe_arr = os.listdir(self.bpe_path)
        bpe_arr = [os.path.join(self.bpe_path, f) for f in bpe_arr]
        processor.main(self.org_corpus_dir, self.prep_dir, self.prep_base_name, bpe_arr)


if __name__ == "__main__":
    from nmt_trans.utils import conf_parser
    import sys
    conf_f = sys.argv[1]
    t_conf = conf_parser.parse_conf(conf_f)
    t_obj = RawPreprocessor(t_conf)
    t_obj.run()
