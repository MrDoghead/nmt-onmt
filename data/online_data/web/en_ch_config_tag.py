# coding=utf8

from hb_translate.utils import file_helper

config_dict = {
    "data": "/data/translate/hb_trans/data_bin/all_tagged_en_zh",
    "path": "/data/translate/hb_trans/DYCONV_MODEL_en_zh_tagged/checkpoint_last10_avg.pt",
    "beam": 8,
    "remove-bpe": None,
    "source-lang": "en",
    "target-lang": "zh",
    "tokenizer": "moses",
    "moses-no-dash-splits": None,
    "bpe": "subword_nmt",
    "bpe-codes": "/data/translate/hb_trans/prep_all_v1_tagged/code.en",
    "max-tokens": 3072,
}

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
