# coding=utf8

from nmt_trans.utils import file_helper, conf_parser
from nmt_trans.predictor import Predictor
from nmt_trans.tokenizer.basic_tokenizer import BaseTokenizer
import sacrebleu
import os
import sys
from tqdm import tqdm


def main(conf_path, test_dir):
    conf = conf_parser.parse_conf(conf_path)
    dict_path = file_helper.get_online_data("caijing_clean.csv")
    predictor = Predictor(conf, dict_path)

    test_zh_path, test_en_path = [os.path.join(test_dir, name)
                                  for name in ['test.zh', 'test.en']]

    with open(test_zh_path) as f_t_z, \
            open(test_en_path) as f_t_e:
        test_zh = f_t_z.readlines()
        test_en = f_t_e.readlines()

    bs = 1
    if 'speed_baseline' not in sys.argv:
        bs = 16
        # 按zh长度粗排序
        test_zh, test_en = \
            zip(
                *(sorted(zip(test_zh, test_en),
                         key=lambda x: len(x[0])))
            )

    start = 0
    pred_zh = []
    for start in tqdm(range(0, len(test_zh), bs)):
        print(f"start is {start}")
        if start + bs > 1000:
            print(test_en[start:start+bs])
        tmp = predictor.predict(test_en[start:start + bs])
        print(f"cur_len of cur_start: {len(tmp)}")
        pred_zh.extend(tmp)
        # print('\n\n'.join(tmp))
        print(f"current len of pred is: {len(pred_zh)}")
    print(len(pred_zh))
    print(len(test_zh))

    pred_en_path = 'test.pred.en'

    with open(pred_en_path, 'w') as f_p_e:
        f_p_e.write('\n'.join(pred_zh))

    bleu(test_zh, pred_zh[:len(test_zh)])


def bleu(gt, pred):
    gt = _split_texts(gt)
    pred = _split_texts(pred)
    res = sacrebleu.corpus_bleu(pred, [gt])
    print(res.format())


def _split_texts(text_arr):
    tokenizer = BaseTokenizer()
    res = []
    for text in text_arr:
        words = tokenizer.tokenize(text)
        res.append(" ".join(words))
    return res


if __name__ == '__main__':
    conf_path = sys.argv[1]  # config 文件地址
    test_dir = sys.argv[2]  # 要测试文件地址
    main(conf_path, test_dir)
