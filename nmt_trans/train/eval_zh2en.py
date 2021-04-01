#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nmt_trans.utils import file_helper, conf_parser
from nmt_trans.predictor_zh2en import Predictor
import sacrebleu
import os
import sys

def main(conf_path):
    conf = conf_parser.parse_conf(conf_path)
    dict_path = file_helper.get_online_data("caijing_clean.csv")
    predictor = Predictor(conf, dict_path)
    
    test_zh_path, test_en_path = \
        [os.path.join('/data/tao.hu/haitong_test', name)
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
    pred_en = []
    while (start < len(test_zh)):
        tmp, _ = predictor.predict(test_zh[start:start+bs])
        pred_en.extend(tmp)
        start += bs
        print('\n\n'.join(tmp))
    
    pred_en_path = 'test.pred.en'

    with open(pred_en_path, 'w') as f_p_e:
        f_p_e.write('\n'.join(pred_en))
    
    bleu(test_en, pred_en)


def bleu(gt, pred):
    res = sacrebleu.corpus_bleu(gt, [pred])
    print(res.format())


if __name__ == '__main__':
    conf_path = sys.argv[1]
    main(conf_path)

