# coding=utf8

import re
import numpy as np
import os
from tqdm import tqdm

from nmt_trans.utils import custom_tag

non_space_pat = r'[(（]+\s*[)）]+'


class PrecessTag(object):
    def __init__(self, dict_path):
        self.tagger = custom_tag.TagHelper(dict_path)

    def process_en(self, in_dir, out_dir, base_name):
        """
        本函数将原始的语料处理成中英
        """
        os.makedirs(out_dir, exist_ok=True)
        langs = ["en", "zh"]
        org_files = [os.path.join(in_dir, base_name + "." + lang) for lang in langs]
        out_files = [os.path.join(out_dir, base_name + "." + lang) for lang in langs]
        with open(org_files[0]) as in_en:
            with open(org_files[1]) as in_zh:
                with open(out_files[0], "w") as out_en:
                    with open(out_files[1], "w") as out_zh:
                        for en_line in tqdm(in_en):
                            zh_line = in_zh.readline()
                            new_en, new_zh = self.tag_en(en_line, zh_line)
                            out_en.write(new_en)
                            out_zh.write(new_zh)

    def tag_en(self, en_line, zh_line):
        en_tags = self.tagger.search_en(en_line)
        if len(en_tags) < 1:
            return en_line, zh_line
        en_words = [en_line[s:e+1] for s, e in en_tags]
        cnt_start = np.random.randint(20)
        en_word2num = words2dict(en_words, cnt_start)
        zh_tags = self.tagger.search_zh(zh_line)
        if len(zh_tags) < 1:
            return en_line, zh_line
        zh_word_pos_dict = self._get_word_pos_dict(zh_line, zh_tags)
        f_en_pos_2_tag = {}
        f_zh_pos_2_tag = {}
        rep_en_words = set()
        for i, en_word in enumerate(en_words):
            tmp_zh_word = self.tagger.get_zh(en_word)
            if tmp_zh_word is not None and tmp_zh_word in zh_word_pos_dict:
                rep_en_words.add(en_word)
                s, e = en_tags[i]
                cur_en_pos = en_word2num.get(en_word)
                f_en_pos_2_tag[(s, e+1)] = (en_word, cur_en_pos)
                for tmp_zh_pos in zh_word_pos_dict[tmp_zh_word]:
                    f_zh_pos_2_tag[tmp_zh_pos] = (tmp_zh_word, cur_en_pos)
        new_enline = self.tagger.replace_line(en_line, f_en_pos_2_tag)
        new_zhline = self.tagger.replace_line(zh_line, f_zh_pos_2_tag)
        new_zhline = self._remove_rep_words(new_zhline, rep_en_words)
        return new_enline, new_zhline

    def _remove_rep_words(self, sen, rep_words):
        for word in rep_words:
            sen = sen.replace(word, "")
        sen = re.sub(non_space_pat, "", sen)
        return sen

    def _get_word_pos_dict(self, sen, pos_list):
        res = dict()
        for s, e in pos_list:
            key = sen[s:e+1]
            if key not in res:
                res[key] = []
            res[key].append((s, e+1))
        return res


def words2dict(word_list, start=0):
    res_dict = {}
    cnt = start
    for word in word_list:
        if word not in res_dict:
            res_dict[word] = cnt
            cnt += 1
    return res_dict


if __name__ == "__main__":
    # for testing data
    from nmt_trans.utils import file_helper
    test_f = file_helper.get_online_data("caijing_clean.csv")
    tag_helper = PrecessTag(test_f)
    test_str = "UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier. More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year. U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year."
    test_zh = "日经指数早先报道称，去年，为了赶上高通( Qualcomm )和联发科技( MediaTek ) ，该公司加快了5G芯片开发的步伐。最近，这家中国移动芯片开发商从中国国有集成电路基金（简称“大额基金” ）获得了45亿元人民币（合6.3亿美元）的资金，并准备于今年晚些时候在中国版的纳斯达克( Nasdaq ) — —科创板上市。自去年5月16日以来，总部位于美国的高通( Qualcomm )不得不获得美国商务部( Department of Commerce )的许可，才能为华为供货。"
    new_en, new_zh = tag_helper.tag_en(test_str, test_zh)
    print(new_en)
    print(new_zh)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str)
    # parser.add_argument('--output', type=str)
    # parser.add_argument('--b_name', type=str)
    # args = parser.parse_args()
    # tag_helper.process_en(args.input, args.output, args.b_name)
