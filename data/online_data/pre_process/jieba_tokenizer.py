# coding=utf8

import re
import jieba


class Tokenizer(object):
    def __init__(self, dict_path=None):
        self.sp_pat = re.compile(r"(<ent>|<\\ent>|｟\d+｠)")
        # if dict_path is not None:
        #     self.sp_pat = self._get_regex(dict_path)
        # 由于jieba 是lazy init， 这里调用init完成切分
        jieba.cut("你好，世界")

    def _get_regex(self, f_path):
        tmp_arr = []
        with open(f_path) as in_:
            for ln in in_:
                ln = ln.strip()
                tmp_arr.append(ln)
        res = "(" + "|".join(tmp_arr) + ")"
        return re.compile(res)

    def tokenize(self, sen):
        if self.sp_pat is not None:
            res = []
            blocks = re.split(self.sp_pat, sen)
            for block in blocks:
                if self.sp_pat.match(block):
                    res.append(block)
                else:
                    res.extend(jieba.cut(block))
            return res
        return list(jieba.cut(sen))


class DeTokenizer(object):
    def __init__(self):
        pass

    def detokenize(self, word_list):
        return "".join(word_list)


if __name__ == '__main__':
    test_str = "往返训练的流程是这样子的，假设我们有f单语言数据，那么先用正向模型翻译成e'，" \
               "之后用这个e‘和f来训练反向模型。当要用e的单语言数据，只要反过来操作就好了。"
    test_str = "recently ｟1｠ cause some people died"

    _dict_path = "../online_data/jieba_dict"
    tokenizer = Tokenizer(_dict_path)
    t_res = tokenizer.tokenize(test_str)
    print("\t".join(t_res))
