# coding=utf8

from collections import OrderedDict
from hb_translate.utils import file_helper

config_list = [
    ("data", file_helper.get_online_data("data_bin/fq_1kw_en_zh_v2")),
    ("path", file_helper.get_online_data("model_bin/fq_1kw_en_zh_conv_v2/checkpoint_best.pt")),
    ("beam", 4),
    ("remove-bpe", None),
    ("source-lang", "en"),
    ("target-lang", "zh"),
    ("tokenizer", "moses"),
    ("moses-no-dash-splits", None),
    ("bpe", "subword_nmt"),
    ("bpe-codes", file_helper.get_online_data("bpe/code.en")),
    ("max-tokens", 3072),
    ("skip-invalid-size-inputs-valid-test", None),
]

config_dict = OrderedDict(config_list)

pos_type = {"data"}

align_dict_path = file_helper.get_online_data('caijing_clean.csv')


def get_args_arr():
    res = []
    for key, val in config_dict.items():
        if key not in pos_type:
            res.append("--" + key)
        
        if val is not None:
            res.append(str(val))
    return res


if __name__ == "__main__":
    print(get_args_arr())
    print(" ".join(get_args_arr()))
