# coding=utf8

from onmt.bin import translate
from hb_chat.utils import file_helper, conf_parser


class NMTEvaluator(object):
    def __init__(self, conf):
        self.conf = conf

    def get_args(self, parser):
        src_path = file_helper.get_real_path(self.conf.src_test_path)
        dst_path = file_helper.get_real_path(self.conf.dst_test_path)
        model_path = file_helper.get_abs_path(self.conf.model_path) + self.conf.eval_info.model_suffix
        out_path = file_helper.get_abs_path(self.conf.eval_info.output)
        file_helper.mk_folder_for_file(out_path)
        args_dict = {
            "src": src_path,
            "tgt": dst_path,
            "model": model_path,
            "output": out_path,
        }
        conf_dict = conf_parser.conf2dict(self.conf.eval_info)
        for key, val in conf_dict.items():
            if key not in args_dict:
                args_dict[key] = val
        args_arr = conf_parser.dict2args(args_dict)
        args, _ = parser.parse_known_args(args_arr)
        return args

    def run(self):
        parser = translate._get_parser()
        args = self.get_args(parser)
        translate.translate(args)


def main():
    t_conf_path = file_helper.get_conf_file("chat_config.json")
    conf = conf_parser.parse_conf(t_conf_path)
    t_trainer = NMTEvaluator(conf)
    t_trainer.run()


if __name__ == "__main__":
    main()
