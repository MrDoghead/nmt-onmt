# coding=utf8

import os
import re
import ctranslate2
from subword_nmt import apply_bpe
from nmt_trans.utils import file_helper, conf_parser
from nmt_trans.tokenizer import jieba_tokenizer, mose_tokenizer
from nmt_trans.utils.trans_dict_manager import ZhEnMapper
from nmt_trans.tags import (
    custom_tag,
    num_tag,
    date_tag,
    price_tag,
    amount_tag,
)
from nmt_trans.utils.str_utils import split_sub_sen, strQ2B
from nmt_trans.wrapper import menu_wrapper
from nmt_trans.wrapper import multi_tagger_wrapper

no_space_ch_pat = re.compile(r'[\u4e00-\u9fa5<>\\()（）]+')
ent_pat = re.compile(r'<ent>(.+?)<\\ent>')
uneed_pat = re.compile(r'\([0-9a-zA-z\s]+\)|>\s+')
noise_pat = re.compile(r'<ent>[\d\s]+<\\ent>|>[\d\s]+<\\ent>|<ent>[\d\s]+<|<ent>|<\\ent>|<|>')


class Predictor(object):
    def __init__(self, conf, cus_dict):
        self.conf = conf
        model_path = file_helper.get_real_path(self.conf.pred_info.c_model_path)
        self.translator = ctranslate2.Translator(
            model_path,
            device="auto",
            intra_threads=self.conf.pred_info.c_translate_thread
        )
        self.bpe = self._load_bpe()
        self.bpe_symbol = "@@ "
        self.tokenizer = mose_tokenizer.MTokenizer("en")
        self.detokenizer = mose_tokenizer.MDTokenizer("en")
        self.jb_tokenizer = jieba_tokenizer.Tokenizer()
        self.tag_helpers = [
            custom_tag.TagHelper(cus_dict),
            num_tag.TagHelper(),
            date_tag.TagHelper(),
            price_tag.TagHelper(),
            amount_tag.TagHelper(),
        ]
        self.zh_en_mapper = ZhEnMapper()
        self.wrapper = menu_wrapper.MenuWrapper

    def _load_bpe(self):
        bpe_path = file_helper.get_real_path(self.conf.bpe_code_path)
        bpe_path = os.path.join(bpe_path, "code.zh")
        with open(bpe_path) as in_:
            bpe = apply_bpe.BPE(in_)
            return bpe

    def _prep_sen(self, sen):
        """
        :param sen:
        :return: 分好词的列表
        """
        sen = sen.strip()
        f_str = self.encode_fn(sen)
        f_words = f_str.split()
        return f_words

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.jb_tokenizer.tokenize(x)
            x = " ".join(x)
        if self.bpe is not None:
            x = self.bpe.process_line(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self._decode_bpe(x)
        return self.detokenizer.detokenize(x.split())

    def _post_pro_sen(self, word_list):
        word_list = word_list[0]["tokens"]
        ans_str = " ".join(word_list)
        ans_str = self.decode_fn(ans_str)
        # ans_str = str_utils.remove_repeat(ans_str)
        return ans_str

    def _is_no_space_ch(self, x):
        return re.match(no_space_ch_pat, x)

    def _decode_bpe(self, x: str) -> str:
        return (x + ' ').replace(self.bpe_symbol, '').rstrip() 

    def _predict_impl(self, input_sens):
        emp_idxes = set()
        map_idxes = dict()
        for i, sen in enumerate(input_sens):
            if len(sen.strip()) == 0:
                emp_idxes.add(i)
            else:
                zh_sen = self.zh_en_mapper.get_trans(sen.strip())
                if zh_sen is not None:
                    map_idxes[i] = zh_sen

        inputs, tag_dicts = self.pre_process(input_sens)
        inputs, m_wraper = multi_tagger_wrapper.MultiTagWrapper.pre_process(inputs)
        # append EOS to the end to adapt the fairseq model
        input_list = [self._prep_sen(sen) + ['</s>'] for sen in inputs]
        res_list = self.translator.translate_batch(
            input_list,
            beam_size=8,
            num_hypotheses=1,
            length_penalty=1.1,
            max_decoding_length=250
        )

        f_result = [self._post_pro_sen(word_list) for word_list in res_list]
        f_result = multi_tagger_wrapper.MultiTagWrapper.post_process(f_result, m_wraper)
        ff_result = []
        # states 表示decode的替换是否可能会出错
        states = []
        for i, out_sen in enumerate(f_result):
            if i in emp_idxes:
                ff_result.append(" ")
                states.append(True)
            elif i in map_idxes:
                ff_result.append(map_idxes[i])
                states.append(True)
            else:
                temp_states = True
                for j, tag_helper in enumerate(self.tag_helpers):
                    out_sen, is_correct = tag_helper.decode_en(out_sen, tag_dicts[j][i])
                    temp_states = temp_states and is_correct
                ff_result.append(out_sen)
                states.append(temp_states)

        ff_result = self.final_proc(ff_result)
        return ff_result, states

    def pre_process(self, sens):
        res_sen = []
        res_dicts = [[] for _ in range(len(self.tag_helpers))]
        for sen in sens:
            for i, tag_helper in enumerate(self.tag_helpers):
                sen, out_dict = tag_helper.encode_zh(sen)
                res_dicts[i].append(out_dict)
            res_sen.append(sen)
        return res_sen, res_dicts

    def update_tag_words(self, mod_dict):
        self.tag_helpers[0].update(mod_dict)

    def update_tag_batchly(self):
        self.tag_helpers[0].update_batchly()

    def final_proc(self, sens):
        res = []
        for sen in sens:
            res.append(self._repl_with_dict(sen))
        return res

    def _repl_with_dict(self, sen):
        words = self.jb_tokenizer.tokenize(sen)
        res = []
        for word in words:
            rep_word = self.zh_en_mapper.get_trans(word)
            if rep_word is None:
                res.append(word)
            else:
                res.append(rep_word)
        new_sen = "".join(res)
        new_sen = self.decode_fn(new_sen)
        return new_sen

#    def _split_sub_sen(self, sen):
#        """
#        用来切分句子， 当句子长度超过150个单词的时候，
#        """
#        thres = 100
#        if len(sen.strip().split()) < thres:
#            return [sen]
#        res = split_sub_sen(sen)
#        return res
#
#    def split_sub_sens(self, sens):
#        """
#        将句子全部切分出来， 主要用来切分容易过长的句子；
#        """
#        begin = 0
#        sen_no_map = dict()
#        sub_sens = []
#        for i, sen in enumerate(sens):
#            cur_sub_sens = self._split_sub_sen(sen)
#            if cur_sub_sens is not None:
#                sub_sens.extend(cur_sub_sens)
#            for j in range(len(cur_sub_sens)):
#                sen_no_map[begin+j] = i
#                begin += len(cur_sub_sens)
#        return sub_sens, sen_no_map
#
#    def merge_sub_sens(self, sub_sens, sen_no_map):
#        mid_sens = [""] * len(sen_no_map)
#        for i, sub_sen in enumerate(sub_sens):
#            if i in sen_no_map:
#                j = sen_no_map[i]
#                mid_sens[j] = mid_sens[j] + sub_sen
#        return mid_sens

    def predict(self, input_sens):
        # 多个空格替换成一个
        space_pat = re.compile(r'\s+')
        input_sens = [space_pat.sub(' ', sen) for sen in input_sens]
        
        #input_sens, wrap_info = self.wrapper.pre_process(input_sens)
#        sub_sens, sen_no_map = self.split_sub_sens(input_sens)
#        trans_res, sub_states = self._predict_impl(sub_sens)
        res, states = self._predict_impl(input_sens)
#        res, states = self.merge_sub_sens(trans_res, sen_no_map)
        #out_res, out_states = self.wrapper.post_process_with_states(res, states, wrap_info)
        
        # 英文全角转半角
        #out_res = strQ2B(out_res)
        out_res = strQ2B(res)
        # / 边上的空格 去掉
        out_res = [sen.replace(' / ', '/').replace(' & ', '&') for sen in out_res]
        #return out_res, out_states
        return out_res


def test(sens, conf_name):
    conf_path = file_helper.get_conf_file(conf_name)
    conf = conf_parser.parse_conf(conf_path)
    dict_path = file_helper.get_online_data("caijing_clean.csv")
    predictor = Predictor(conf, dict_path)
    res = predictor.predict(sens)
    for sen in res:
        print(sen)


if __name__ == "__main__":
    import sys
    conf_path = sys.argv[1]
    #test_sen = ["你好世界", "我不这么认为", "你有男朋友吗？ ", "这病有药治疗吗?"]
    test_sen = ["1994 年 5 月 17 日 安全 理事会 第 3@@ 377 次 会议 通过"]
    test(test_sen, conf_path)
