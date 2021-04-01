# coding=utf8

import os
from onmt.bin import preprocess
from nmt_trans.utils import file_helper, conf_parser


class NMTProcessor(object):
    def __init__(self, conf):
        self.conf = conf
        self.save_path = file_helper.get_abs_path(self.conf.train_info.train_fmt_path)
        file_helper.mk_folder_for_file(self.save_path)
        self.prep_dir = self.conf.preped_dir
        self.vocab_path = file_helper.get_abs_path(self.conf.train_info.vocab_path)

    def get_args(self, parser):
        args_dict = {
            "train_src": file_helper.get_real_path(os.path.join(self.prep_dir, self.conf.src_train_path)),
            "train_tgt": file_helper.get_real_path(os.path.join(self.prep_dir, self.conf.dst_train_path)),
            "valid_src": file_helper.get_real_path(os.path.join(self.prep_dir, self.conf.src_test_path)),
            "valid_tgt": file_helper.get_real_path(os.path.join(self.prep_dir, self.conf.dst_test_path)),
            "save_data": self.save_path,
            "overwrite": None,
            "src_seq_length": self.conf.train_info.src_seq_length,
            "tgt_seq_length": self.conf.train_info.tgt_seq_length,
            "filter_valid": None,
            "src_vocab": self.vocab_path,
            "dynamic_dict": None,
            "share_vocab ": None
        }
        args_arr = conf_parser.dict2args(args_dict)
        args = parser.parse_known_args(args_arr)
        return args

    def run(self):
        parser = preprocess._get_parser()
        args, _ = self.get_args(parser)
        preprocess.preprocess(args)


if __name__ == "__main__":
    import sys
    # t_conf_path = file_helper.get_conf_file("chat_config.json")
    t_conf_path = sys.argv[1]
    conf = conf_parser.parse_conf(t_conf_path)
    t_processor = NMTProcessor(conf)
    t_processor.run()

