# coding=utf8

import re
import ctranslate2
from subword_nmt import apply_bpe
from hb_chat.utils import file_helper, conf_parser, text_util
from hb_chat.tokenizer import jieba_tokenizer

no_space_ch_pat = re.compile(r'[\u4e00-\u9fa5<>\\()（）]+')


class Predictor(object):
    def __init__(self, conf):
        self.conf = conf
        model_path = file_helper.get_real_path(self.conf.pred_info.c_model_path)
        self.translator = ctranslate2.Translator(model_path)
        self.tokenizer = jieba_tokenizer.Tokenizer()
        self.bpe = self._load_bpe()
        self.bpe_symbol = "@@ "

    def _load_bpe(self):
        bpe_path = file_helper.get_real_path(self.conf.bpe_code_path)
        with open(bpe_path) as in_:
            bpe = apply_bpe.BPE(in_)
            return bpe

    def predict(self, sen_list):
        input_list = [self._prep_sen(sen) for sen in sen_list]
        res_list = self.translator.translate_batch(input_list)
        res = [self._post_pro_sen(word_list) for word_list in res_list]
        return res

    def _prep_sen(self, sen):
        """
        :param sen:
        :return:
        """
        sen = sen.strip()
        rub_pat = re.compile(r'[\s=。〜*/〒_～><|^O()【】（）￣︶#]+')
        sen = re.sub(rub_pat, " ", sen)
        words = self.tokenizer.tokenize(sen)
        words_str = " ".join(words)
        f_str = self.bpe.process_line(words_str)
        f_words = f_str.split()
        return f_words

    def _post_pro_sen(self, word_list):
        word_list = word_list[0]["tokens"]
        ans_str = " ".join(word_list)
        ans_str = self.detoken(ans_str)
        ans_str = text_util.remove_repeat(ans_str)
        return ans_str

    def detoken(self, x):
        if self.bpe is not None:
            x = self.decode_bpe(x)
        word_list = x.split()
        _need_sp = False
        result = []
        for word in word_list:
            if not self._is_no_space_ch(word):
                if _need_sp:
                    result.append(" ")
                _need_sp = True
            else:
                _need_sp = False
            result.append(word)
        return "".join(result)

    def decode_bpe(self, x: str) -> str:
        return (x + ' ').replace(self.bpe_symbol, '').rstrip()

    def _is_no_space_ch(self, x):
        return re.match(no_space_ch_pat, x)


def test(sens):
    conf_path = file_helper.get_conf_file("chat_config.json")
    conf = conf_parser.parse_conf(conf_path)
    predictor = Predictor(conf)
    res = predictor.predict(sens)
    for sen in res:
        print(sen)


if __name__ == "__main__":
    test_sen = ["你好", "今天天气如何", "你又有男朋友吗？ ", "这病有药治疗吗?"]
    test(test_sen)
