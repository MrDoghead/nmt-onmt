# coding=utf8

import os
import shutil
import torch
from fairseq_cli import train
from fairseq import options
from fairseq import distributed_utils
from nmt_trans.utils import file_helper, conf_parser


class Trainer(object):
    def __init__(self, conf):
        self.conf = conf
        self.input_data = file_helper.get_abs_path(self.conf.preprocess.destdir)
        self.out_path = file_helper.get_abs_path(self.conf.train.save_dir)
        file_helper.mk_folder_for_file(self.out_path)
        if os.path.exists(self.out_path):
            shutil.rmtree(self.out_path)

    def get_args(self, parser):
        args_dict = {
            "save-dir": self.out_path,
        }
        conf_dict = conf_parser.conf2dict(self.conf.train)
        conf_dict.pop("save_dir", None)
        args_dict.update(conf_dict)
        args_arr = conf_parser.dict2args(args_dict)
        f_args_arr = [self.input_data]
        f_args_arr.extend(args_arr)
        args = options.parse_args_and_arch(parser, input_args=f_args_arr)
        return args

    def train(self):
        parser = options.get_training_parser()
        args = self.get_args(parser)
        cfg = train.convert_namespace_to_omegaconf(args)
        if args.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, train.main)
        else:
            distributed_utils.call_main(cfg, train.main)


if __name__ == "__main__":
    import sys
    t_conf_path = sys.argv[1]
    # t_conf_path = file_helper.get_conf_file("chat_config.json")
    conf = conf_parser.parse_conf(t_conf_path)
    fq_conf = conf.fairseq_conf
    t_trainer = Trainer(fq_conf)
    t_trainer.train()
