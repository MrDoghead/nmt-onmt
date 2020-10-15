# coding=utf8

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from subword_nmt import learn_bpe, apply_bpe
from hb_chat.tokenizer import jieba_tokenizer
from hb_chat.utils import file_helper

BPE_CODES_NUM = 5000


class DataPreparer(object):
    def __init__(self, conf):
        self.conf = conf
        self.tokenizer = jieba_tokenizer.Tokenizer()
        # self.detokenizer = jieba_tokenizer.DeTokenizer()

    def run(self):
        train_df, test_df = self.read_split()
        self.prepare_df(train_df, is_train=True)
        self.prepare_df(test_df, is_train=False)

    def read_split(self):
        in_path = file_helper.get_real_path(self.conf.raw_tot_data)
        df = pd.read_csv(in_path)
        train_df, test_df = train_test_split(df, test_size=0.05, random_state=73)
        return train_df, test_df

    def _tok_and_save(self, df, sign):
        if sign:  # now is train_path
            src_path = file_helper.get_abs_path(self.conf.tmp_src_train_path)
            dst_path = file_helper.get_abs_path(self.conf.tmp_dst_train_path)
        else:
            src_path = file_helper.get_abs_path(self.conf.tmp_src_test_path)
            dst_path = file_helper.get_abs_path(self.conf.tmp_dst_test_path)
        file_helper.mk_folder_for_file(src_path)
        self._tok_and_save_impl(df['ques'], src_path)
        self._tok_and_save_impl(df['ans'], dst_path)
        if sign:
            self._cons_bpe([src_path, dst_path])
        return src_path, dst_path

    def _tok_and_save_impl(self, sen_arr, src_path):
        with open(src_path, "w") as out_:
            for line in tqdm(sen_arr):
                line = line.strip()
                word_arr = self.tokenizer.tokenize(line)
                out_.write(" ".join(word_arr) + "\n")

    def _cons_bpe(self, f_list):
        tot_train_path = file_helper.get_abs_path(self.conf.tmp_tot_word_path)
        self._merge_path(f_list, tot_train_path)
        bpe_path = file_helper.get_abs_path(self.conf.bpe_code_path)
        file_helper.mk_folder_for_file(bpe_path)
        with open(tot_train_path) as in_:
            with open(bpe_path, "w") as out_:
                learn_bpe.learn_bpe(in_, out_, BPE_CODES_NUM)

    def _merge_path(self, f_list, dst_path):
        file_helper.mk_folder_for_file(dst_path)
        with open(dst_path, "w") as out_:
            for f_path in f_list:
                with open(f_path) as in_:
                    for line in in_:
                        out_.write(line)

    def prepare_df(self, df, is_train):
        """
        :param df: 输入data_frame
        :param is_train: 标识是否是train
        :return:
        """
        src_word_path, dst_word_path = self._tok_and_save(df, is_train)
        if is_train:
            o_src_path = file_helper.get_abs_path(self.conf.src_train_path)
            o_dst_path = file_helper.get_abs_path(self.conf.dst_train_path)
        else:
            o_src_path = file_helper.get_abs_path(self.conf.src_test_path)
            o_dst_path = file_helper.get_abs_path(self.conf.dst_test_path)
        self._apply_bpe(src_word_path, o_src_path)
        self._apply_bpe(dst_word_path, o_dst_path)

    def _apply_bpe(self, word_path, out_path):
        bpe_path = file_helper.get_real_path(self.conf.bpe_code_path)
        with open(bpe_path) as f_code:
            bpe = apply_bpe.BPE(f_code)
            with open(out_path, "w") as out_:
                with open(word_path) as in_:
                    for line in tqdm(in_, "applying bpe"):
                        res_line = bpe.process_line(line)
                        out_.write(res_line)


def main(conf):
    data_preper = DataPreparer(conf)
    data_preper.run()


if __name__ == "__main__":
    conf_path = file_helper.get_conf_file("chat_config.json")
    from hb_chat.utils import conf_parser
    t_conf = conf_parser.parse_conf(conf_path)
    main(t_conf)
