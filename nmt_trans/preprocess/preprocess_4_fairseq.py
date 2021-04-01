# coding=utf8

import os
from fairseq import options
from fairseq_cli import preprocess
from nmt_trans.utils import file_helper, conf_parser


class FairseqPrep(object):
    def __init__(self, conf):
        self.conf = conf
        self.save_path = file_helper.get_abs_path(self.conf.destdir)
        os.makedirs(self.save_path, exist_ok=True)

    def get_args(self, parser):
        args_dict = {
            "trainpref": file_helper.get_abs_path(self.conf.trainpref),
            "validpref": file_helper.get_abs_path(self.conf.validpref),
            "testpref": file_helper.get_abs_path(self.conf.testpref),
            "destdir":  file_helper.get_abs_path(self.conf.destdir),
        }
        conf_dict = conf_parser.conf2dict(self.conf)
        conf_dict.update(args_dict)
        args_arr = conf_parser.dict2args(conf_dict)
        args = parser.parse_known_args(args_arr)
        return args

    def run(self):
        parser = options.get_preprocessing_parser()
        args, _ = self.get_args(parser)
        preprocess.main(args)


if __name__ == "__main__":
    import sys
    t_conf_path = sys.argv[1]
    conf = conf_parser.parse_conf(t_conf_path)
    prep_conf = conf.fairseq_conf.preprocess
    t_processor = FairseqPrep(prep_conf)
    t_processor.run()
