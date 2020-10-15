# coding=utf8

from hb_translate.utils import file_helper

config_dict = {
    "data": "/data/translate/hb_trans/data_bin/fq_1kw_en_zh_v2",
    "path": "/data/translate/hb_trans/model_bin/fq_transformer_en_zh/checkpoint_best.pt",
    "beam": 8,
    "remove-bpe": None,
    "source-lang": "en",
    "target-lang": "zh",
    "tokenizer": "moses",
    "moses-no-dash-splits": None,
    "bpe": "subword_nmt",
    "bpe-codes": "/data/translate/hb_trans/prep_corpus/prep_1kw_tagged_v2/code.en",
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

