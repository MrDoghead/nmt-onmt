config_dict = {
    "data": "/data/translate/hb_trans/data_bin/ft_chinese_en_zh",
    "path": "/data/translate/hb_trans/DYCONV_MODEL_en_zh/checkpoint_best.pt",
    "beam": 8,
    "remove-bpe": None,
    "source-lang": "en",
    "target-lang": "zh",
    "tokenizer": "moses",
    "bpe": "subword_nmt",
    "bpe-codes": "/data/translate/hb_trans/prep_ft_chinese/code",
}

pos_type = {"data"}


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
