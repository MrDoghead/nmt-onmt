# coding=utf8

from sacremoses import MosesTokenizer, MosesDetokenizer


class MTokenizer(object):
    def __init__(self, lang, protected_patterns=["<ent>", "<N>", r"｟.?\d+｠"], not_split_dash=False):
        self.tokenizer = MosesTokenizer(lang=lang)
        self.protected_patterns = protected_patterns
        self.split_dash = not_split_dash

    def tokenize(self, sen, kp_str=True):
        return self.tokenizer.tokenize(sen, return_str=kp_str, protected_patterns=self.protected_patterns)


class MDTokenizer(object):
    def __init__(self, lang):
        self.dt = MosesDetokenizer(lang)

    def detokenize(self, word_list):
        return self.dt.detokenize(word_list)


if __name__ == "__main__":
    _dict_path = "../online_data/mose-dict.txt"
    test_toker = MTokenizer("en")
    test_str = "recently ｟20｠ cause ｟k25｠ some ｟n30｠ people died,//"
    print(test_toker.tokenize(test_str))
