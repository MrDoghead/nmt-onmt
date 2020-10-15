# coding=utf8

import os
import re
import pandas as pd
from tqdm import tqdm
from hb_chat.utils import file_helper
from hb_chat.utils import data_util


class PreProcessor(object):
    def __init__(self, conf):
        self.in_dir = file_helper.get_real_path(conf.raw_data_dir)
        self.out_path = file_helper.get_abs_path(conf.raw_tot_data)
        file_helper.mk_folder_for_file(self.out_path)

    def run(self):
        f_names = os.listdir(self.in_dir)
        res = {
            "ques": [],
            "ans": []
        }
        visited_q_set = set()
        for f_name in f_names:
            if f_name.endswith(".tsv"):
                df = pd.read_csv(os.path.join(self.in_dir, f_name), sep="\t", names=["q", "a"])
                q_arr = df["q"]
                a_arr = df["a"]

                for i, sen in tqdm(enumerate(q_arr)):
                    if not isinstance(sen, str) or not isinstance(a_arr[i], str):
                        continue
                    sen = remove_rub(sen)
                    ans = remove_rub(a_arr[i])
                    if len(sen) < 1 or len(ans) < 1:
                        continue
                    sen = norm_sen(sen)
                    sen_code = data_util.str2md5(sen)
                    if sen_code not in visited_q_set:
                        visited_q_set.add(sen_code)
                        res["ques"].append(sen)
                        res["ans"].append(ans)
        out_df = pd.DataFrame(res)
        out_df.to_csv(self.out_path, index=False)


def remove_rub(sen):
    sen = sen.strip()
    rub_pat = re.compile(r'[\s=。〜*/〒_～><|^O()【】（）￣︶#]+')
    sen = re.sub(rub_pat, "", sen)
    return sen


def norm_sen(sen):
    sen = sen.strip()
    sen_arr = sen.split()
    sen = "".join(sen_arr)
    return sen


if __name__ == "__main__":
    from hb_chat.utils import conf_parser
    test_conf_path = file_helper.get_conf_file("chat_config.json")
    test_conf = conf_parser.parse_conf(test_conf_path)
    test_obj = PreProcessor(test_conf)
    test_obj.run()

