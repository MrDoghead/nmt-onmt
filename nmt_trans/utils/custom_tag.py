# coding=utf8

import re
import ahocorasick
import pandas as pd
import copy
from nmt_trans.utils import data_getter

# data_getter.get_data()


class TagHelper(object):
    def __init__(self, dict_path):
        en_words, zh_words, self.en2zh, self.zh2en = self._read_dict(dict_path)
        self.en_ac = self._build_ac(en_words)
        self.zh_ac = self._build_ac(zh_words)
        self.en_seps = {" ", ".", ",", "!"}
        self.begin_sign = "｟"
        self.end_sign = "｠"
        self.key2method = {
            "add": self.__add_words,
            "mod": self.__modify_words,
            "del": self.__delete_words,
        }
        self.match_pat = re.compile(rf'{self.begin_sign}([\d]+){self.end_sign}')

    def _read_dict(self, f_path):
        df = pd.read_csv(f_path)
        df = df.dropna()
        df = df.reset_index()
        en_parts = df["en"]
        zh_parts = df["cn"]
        en_res = []
        zh_res = []
        en2zh = dict()
        zh2en = dict()
        for i, en in enumerate(en_parts):
            en = self._clean_en(en)
            zh = zh_parts[i]
            zh = self._clean_zh(zh)
            en_res.append(en)
            zh_res.append(zh)
            en2zh[en] = zh
            zh2en[zh] = en
        return en_res, zh_res, en2zh, zh2en

    def _clean_en(self, sen):
        return sen

    def _clean_zh(self, sen):
        return sen

    def _build_ac(self, words):
        ac = ahocorasick.Automaton()
        for i, word in enumerate(words):
            if pd.isna(word):
                continue
            ac.add_word(word, (i, word))
        ac.make_automaton()
        return ac

    def search_en(self, sen):
        result = []
        hit_word_info = self.en_ac.iter(sen)
        for end_index, (insert_order, original_value) in hit_word_info:
            start_index = end_index - len(original_value) + 1
            if self._is_valid_hit(sen, start_index, end_index):
                result.append((start_index, end_index))
        # 去重叠
        if len(result) <= 1:
            return result
        return self._remove_overlap(result)

    def _is_valid_hit(self, sen, start, end):
        if start > 0 and sen[start - 1] not in self.en_seps:
            return False
        if end+1 < len(sen) and sen[end+1] not in self.en_seps:
            return False
        return True

    def search_zh(self, sen):
        result = []
        hit_word_info = self.zh_ac.iter(sen)
        for end_index, (insert_order, original_value) in hit_word_info:
            start_index = end_index - len(original_value) + 1
            result.append((start_index, end_index))
        # 去重叠
        if len(result) <= 1:
            return result
        return self._remove_overlap(result)

    def _remove_overlap(self, input_list):
        res = []
        sorted_list = sorted(input_list, key=lambda x: x[1] - x[0], reverse=True)
        sorted_list = sorted(sorted_list, key=lambda x: x[0])
        for start, end in sorted_list:
            if not self._is_contained(start, end, res):
                res.append([start, end])
        return res

    def _is_contained(self, start, end, tuple_list):
        for pre_star, pre_end in tuple_list:
            if pre_star <= start <= pre_end:
                return True
            if pre_star <= end <= pre_end:
                return True
        return False

    def get_en(self, zh_words):
        return self.zh2en.get(zh_words, None)

    def get_zh(self, en_words):
        return self.en2zh.get(en_words, None)

    def encode_en(self, sen):
        """
        主要用替换来做预处理；
        """
        return self._encode_imp(sen, self.search_en, self.get_zh)

    def encode_zh(self, sen):
        return self._encode_imp(sen, self.search_zh, self.get_en)

    def _encode_imp(self, sen, search_func, word_check_func):
        tags = search_func(sen)
        if len(tags) < 1:
            return sen, {}
        pos2tag = {}
        words = [(sen[s:e + 1], (s, e)) for s, e in tags]
        words = [word for word in words if word_check_func(word[0]) is not None]
        p_words = [word[0] for word in words]
        word2index = words2dict(p_words)
        for word_pair in words:
            word = word_pair[0]
            s, e = word_pair[1]
            pos2tag[(s, e + 1)] = (word, word2index[word])
        new_line = self.replace_line(sen, pos2tag)
        index2word = dict([(v, k) for k, v in word2index.items()])
        return new_line, index2word

    def _get_word_pos_dict(self, sen, pos_list):
        res = dict()
        for s, e in pos_list:
            key = sen[s:e+1]
            if key not in res:
                res[key] = []
            res[key].append((s, e+1))
        return res

    def _split_lines(self, line, pos_list):
        sorted_list = sorted(pos_list, key=lambda x: x[0])
        flat_list = [item for sublist in sorted_list for item in sublist]
        begin = 0
        line_infos = []
        for elem in flat_list:
            key = (begin, elem)
            sub_line = line[begin:elem]
            line_infos.append([key, sub_line])
            begin = elem
        if begin < len(line):
            sub_line = line[begin:]
            key = (begin, len(line))
            line_infos.append([key, sub_line])
        return line_infos

    def replace_line(self, origin_line, pos_word_list):
        line_infos = self._split_lines(origin_line, pos_word_list.keys())
        sub_lines = []
        for key, sub_line in line_infos:
            if key in pos_word_list:
                word_val, idx = pos_word_list[key]
                val_str = f"{self.begin_sign}{idx}{self.end_sign}"
                sub_line = val_str
            sub_lines.append(sub_line)
        return "".join(sub_lines)

    def decode_en(self, sen, id2word):
        return self._decode_imp(sen, id2word, self.get_en)

    def decode_zh(self, sen, id2word):
        return self._decode_imp(sen, id2word, self.get_zh)

    def _decode_imp(self, sen, id2word, word_map_func):
        if len(id2word) < 1:
            return sen
        new_sen = sen
        match_arr = re.finditer(self.match_pat, sen)
        for match in match_arr:
            match_str = match.group(0)
            en_id = int(match.group(1).strip())
            org_word = id2word.get(en_id, None)
            if org_word is not None:
                dest_word = word_map_func(org_word)
                if dest_word is not None:
                    new_sen = new_sen.replace(match_str, dest_word)
        new_sen = new_sen.replace(self.begin_sign, "")
        new_sen = new_sen.replace(self.end_sign, "")
        return new_sen

    def update_batchly(self):
        """
        本代码主要是用来支持对词典的增删改;
        mod_dict 应该有:{"add":[{zh: w1, en:ew1,}, ... ],
                        "mod":[{zh:w1, en:w2}], "del":[{zh: w1, en:ew1,}]}
        """
        data_getter.get_data()
        en_words, zh_words, en2zh, zh2en = self._read_dict(data_getter.dict_path)
        en_ac = self._build_ac(en_words)
        zh_ac = self._build_ac(zh_words)
        # 替换内存
        self.en2zh = en2zh
        self.zh2en = zh2en
        self.en_ac = en_ac
        self.zh_ac = zh_ac
        msg = "sucess"
        return msg

    def update(self, mod_dict):
        """
        本代码主要是用来支持对词典的增删改;
        mod_dict 应该有:{"add":[{zh: w1, en:ew1,}, ... ],
                        "mod":[{zh:w1, en:w2}], "del":[{zh: w1, en:ew1,}]}
        """

        new_en2zh = copy.copy(self.en2zh)
        new_zh2en = copy.copy(self.zh2en)
        for key, val in mod_dict.items():
            if key in self.key2method:
                func = self.key2method[key]
                func(new_en2zh, new_zh2en, val)
        new_en_ac = self._build_ac(new_en2zh.keys())
        new_zh_ac = self._build_ac(new_zh2en.keys())
        # 替换内存
        self.en2zh = new_en2zh
        self.zh2en = new_zh2en
        self.en_ac = new_en_ac
        self.zh_ac = new_zh_ac
        msg = "sucess"
        return msg

    def __add_words(self, en2zh, zh2en, word_list):
        if word_list is None:
            return
        for _dict in word_list:
            if _dict is None:
                continue 
            for lang, word in _dict.items():
                if lang == "en":
                    dest_word = _dict.get("zh", None)
                    if dest_word is not None:
                        en2zh[word] = dest_word
                elif lang == "zh":
                    dest_word = _dict.get("en", None)
                    if dest_word is not None:
                        zh2en[word] = dest_word

    def __modify_words(self, en2zh, zh2en, word_list):
        if word_list is None:
            return
        for _dict in word_list:
            if _dict is None:
                continue
            for lang, word in _dict.items():
                if lang == "en":
                    dest_word = _dict.get("zh", None)
                    if dest_word is not None:
                        en2zh[word] = dest_word
                elif lang == "zh":
                    dest_word = _dict.get("en", None)
                    if dest_word is not None:
                        zh2en[word] = dest_word

    def __delete_words(self, en2zh, zh2en, word_list):
        if word_list is None:
            return
        for _dict in word_list:
            if _dict is None:
                continue
            for lang, word in _dict.items():
                if lang == "en":
                    if word in en2zh:
                        en2zh.pop(word)
                elif lang == "zh":
                    if word in zh2en:
                        zh2en.pop(word)


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
    test_str = "UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier. More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year. U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year."
    test_zh = "日经指数早先报道称，去年，为了赶上高通( Qualcomm )和联发科技( MediaTek ) ，该公司加快了5G芯片开发的步伐。最近，这家中国移动芯片开发商从中国国有集成电路基金（简称“大额基金” ）获得了45亿元人民币（合6.3亿美元）的资金，并准备于今年晚些时候在中国版的纳斯达克( Nasdaq ) — —科创板上市。自去年5月16日以来，总部位于美国的高通( Qualcomm )不得不获得美国商务部( Department of Commerce )的许可，才能为华为供货。"
    tag_helper = TagHelper(test_f)
    new_en = tag_helper.encode_en(test_str)
    new_zh = tag_helper.encode_zh(test_zh)
    print(new_en)
    print(new_zh)
    tag_helper.update({"add": [{"zh": "加快", "en": "accelerated"}]})
    new_en, t_id2dict = tag_helper.encode_en(test_str)
    new_zh, t_id2dict = tag_helper.encode_zh(test_zh)
    print(new_en)
    print(new_zh)
    out_en = tag_helper.decode_en(new_en, t_id2dict)
    print(out_en)
