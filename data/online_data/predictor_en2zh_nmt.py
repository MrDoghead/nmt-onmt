# coding=utf8

import re
import ctranslate2
from subword_nmt import learn_bpe, apply_bpe
from sacremoses import MosesPunctNormalizer
from pre_process import mose_tokenizer
from pre_process import custom_tag
from utils import str_utils

no_space_ch_pat = re.compile(r'[\u4e00-\u9fa5<>\\()（）]+')
ent_pat = re.compile(r'<ent>(.+?)<\\ent>')
uneed_pat = re.compile(r'\([0-9a-zA-z\s]+\)|>\s+')
noise_pat = re.compile(r'<ent>[\d\s]+<\\ent>|>[\d\s]+<\\ent>|<ent>[\d\s]+<|<ent>|<\\ent>|<|>')


class PredictorEn(object):
    def __init__(self, conf, *kargs, **kwargs):
        self.model = ctranslate2.Translator(conf.model_dir)
        self.tokenizer = mose_tokenizer.MTokenizer("en")
        self.bpe = self.build_bpe(conf)
        self.tag_helper = custom_tag.TagHelper(kargs[0])
        self.bpe_symbol = "@@ "
        self.mpn = MosesPunctNormalizer()

    def build_bpe(self, conf):
        in_ = open(conf.bpe_codes)
        bpe = apply_bpe.BPE(in_)
        return bpe

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.tokenizer(x)
        if self.bpe is not None:
            x = self.bpe.process_line(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = (x + ' ').replace(self.bpe_symbol, '').rstrip()
        word_list = x.split()
        _need_sp = False
        result = []
        for word in word_list:
            if not self._is_no_sapce_ch(word):
                if _need_sp:
                    result.append(" ")
                _need_sp = True
            else:
               _need_sp = False
            result.append(word)
        return "".join(result)

    def _is_no_sapce_ch(self, x):
        return re.match(no_space_ch_pat, x)

    def predict(self, input_sens):
        emp_idxes = set()
        for i, sen in enumerate(input_sens):
            if len(sen.strip()) == 0:
                emp_idxes.add(i)

        inputs, en_words_arr = self.pre_process(input_sens)
        inputs = [sen.split() for sen in inputs]
        f_results = self.model.translate_batch(inputs)

        ff_result = []
        for i, out_sen in enumerate(f_results):
            if i in emp_idxes:
                ff_result.append(" ")
            else:
                out_sen = " ".join(out_sen[0]["tokens"])
                out_sen = self.decode_fn(out_sen)
                ff_result.append(self.tag_helper.decode_zh(out_sen, en_words_arr[i]))
        return ff_result

    def pre_process(self, sens):
        res_sen = []
        res_dict = []
        for sen in sens:
            sen = str_utils.preprocess_eng(sen)
            sen = self.mpn.normalize(sen)
            out_sen, out_map = self.tag_helper.encode_en(sen)
            res_sen.append(out_sen)
            res_dict.append(out_map)
        return res_sen, res_dict


if __name__ == "__main__":
    from web import en_ch_nmt_config
    cus_word_path = "online_data/caijing_clean.csv"
    test_predictor = PredictorEn(en_ch_nmt_config, cus_word_path)
    test_str = [
        "Hello, world",
        "UNISOC last year accelerated its 5G chip development to catch up with Qualcomm and MediaTek, Nikkei reported earlier. More recently the Chinese mobile chip developer recently received 4.5 billion yuan ($630 million) from China's national integrated circuit fund, the so-called Big Fund, and is preparing to list on the Shanghai STAR tech board, the Chinese version of Nasdaq, later this year. U.S.-based Qualcomm has had to have a license from the Department of Commerce to supply Huawei since May 16 last year."
    ]
    test_res = test_predictor.predict(test_str)
    for str_ in test_res:
        print(str_)
