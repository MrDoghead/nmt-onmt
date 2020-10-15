# coding=utf8

import pandas as pd


def do_clean(in_file, out_file):
    df = pd.read_csv(in_file)
    df = df.dropna()
    df = df.reset_index()
    en_key = "en"
    zh_key = "cn"
    en_parts = df[en_key]
    zh_parts = df[zh_key]
    res = {en_key: [], zh_key: []}
    for i, en_part in enumerate(en_parts):
        en_part_arr = en_part.split()
        if len(en_part_arr) > 2 or is_all_upper(en_part):
            res[en_key].append(en_part)
            res[zh_key].append(zh_parts[i])
    res_df = pd.DataFrame(res)
    res_df.to_csv(out_file, index=False)


def is_all_upper(_str):
    return _str.upper() == _str


if __name__ == "__main__":
    input_f = "../online_data/caijing.csv"
    output_f = "../online_data/caijing_clean.csv"
    do_clean(input_f, output_f)
