# coding=utf8

from ctranslate2.converters import opennmt_py
from ctranslate2.specs import transformer_spec
from nmt_trans.utils import file_helper, conf_parser


class Converter(object):
    def __init__(self, conf):
        self.conf = conf
        """
        ct2-opennmt-py-converter --model_path ${mode_dir}/mod_en_step_95000.pt --model_spec TransformerBase 
         --output_dir ${mode_dir}/c_trans_mod_en.pt
        """
        self.num_layers = self.conf.train_info.layers
        self.num_heads = self.conf.train_info.heads
        self.model_path = file_helper.get_abs_path(self.conf.model_path) + self.conf.eval_info.model_suffix
        self.quant = getattr(self.conf.eval_info, "quantization", None)

    def get_args(self):
        need_keys = {"model_path", "model_spec", "output_dir"}
        model_path = file_helper.get_abs_path(self.conf.model_path) + self.conf.eval_info.model_suffix
        out_path = file_helper.get_abs_path(self.conf.pred_info.c_model_path)
        file_helper.mk_folder_for_file(out_path)
        args_dict = {
            "model_path": model_path,
            "output_dir": out_path,
        }
        conf_dict = conf_parser.conf2dict(self.conf.pred_info)
        for key, val in conf_dict.items():
            if key not in args_dict and key in need_keys:
                args_dict[key] = val
        args_arr = conf_parser.dict2args(args_dict)
        args_str = " ".join(args_arr)
        return args_str

    def run(self):
        model_spec = transformer_spec.TransformerSpec(self.num_layers, self.num_heads)
        converter = opennmt_py.OpenNMTPyConverter(self.model_path)
        out_path = file_helper.get_abs_path(self.conf.pred_info.c_model_path)
        file_helper.mk_folder_for_file(out_path)
        converter.convert(out_path, model_spec, quantization=self.quant, force=True)


def main(conf_path):
    # t_conf_path = file_helper.get_conf_file("chat_config.json")
    conf = conf_parser.parse_conf(conf_path)
    t_trainer = Converter(conf)
    t_trainer.run()


if __name__ == "__main__":
    import sys
    t_conf_path = sys.argv[1]
    main(t_conf_path)
