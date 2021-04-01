# coding=utf8

import numpy as np
import os
from tqdm import tqdm

from nmt_trans.tags import (
    custom_tag,
    num_tag,
    date_tag,
    price_tag,
    amount_tag,
)


tag_dropout = 0.3

class PrecessTag(object):
    def __init__(self, dict_path):
        self.taggers = [
            custom_tag.TagHelper(dict_path),
            num_tag.TagHelper(),
            date_tag.TagHelper(),
            price_tag.TagHelper(),
            amount_tag.TagHelper(),
        ]

    def process_parallel(self, in_dir, out_dir, base_name):
        """
        本函数将原始的语料处理成中英
        """
        os.makedirs(out_dir, exist_ok=True)
        langs = ["en", "zh"]
        org_files = [os.path.join(in_dir, base_name + "." + lang) for lang in langs]
        out_files = [os.path.join(out_dir, base_name + "." + lang) for lang in langs]
        with open(org_files[0]) as in_en, \
             open(org_files[1]) as in_zh, \
             open(out_files[0], "w") as out_en, \
             open(out_files[1], "w") as out_zh:
            for en_line in tqdm(in_en):
                zh_line = in_zh.readline()
                # 去掉一些非\s的空格字符 
                en_line = ' '.join(en_line.split())
                zh_line = ' '.join(zh_line.split())
                samples = [(en_line, zh_line)]
                if np.random.random() > tag_dropout:
                    for tagger in self.taggers:
                        e, z = samples[0]
                        # 由于存在上采样 可能会tag出多个samples
                        samples = tagger.tag_parallel(e, z)
                        if len(samples) != 1:
                            break
                for e, z in samples:
                    out_en.write(e.strip() + '\n')
                    out_zh.write(z.strip() + '\n')


if __name__ == "__main__":
    # for testing data
    from nmt_trans.utils import file_helper
    test_f = file_helper.get_online_data("caijing_clean.csv")
    tag_helper = PrecessTag(test_f)
#    test_str = "UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier. More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year. U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year." + \
#               "Rose by 5 per cent, 12 %, -3525.222%"
#    test_zh = "日经指数早先报道称，去年，为了赶上高通( Qualcomm )和联发科技( MediaTek ) ，该公司加快了5G芯片开发的步伐。最近，这家中国移动芯片开发商从中国国有集成电路基金（简称“大额基金” ）获得了45亿元人民币（合6.3亿美元）的资金，并准备于今年晚些时候在中国版的纳斯达克( Nasdaq ) — —科创板上市。自去年5月16日以来，总部位于美国的高通( Qualcomm )不得不获得美国商务部( Department of Commerce )的许可，才能为华为供货。" + \
#              "增加5%, 3525.222%"
#    new_en, new_zh = tag_helper.process_parallel(test_str, test_zh)
#    print(new_en)
#    print(new_zh)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str)
    # parser.add_argument('--output', type=str)
    # parser.add_argument('--b_name', type=str)
    # args = parser.parse_args()
    # tag_helper.process_en(args.input, args.output, args.b_name)
