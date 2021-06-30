# coding=utf8

from nmt_trans.utils import file_helper, conf_parser
from nmt_trans.predictor_zh2en import Predictor
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
                                  for name in ['val.zh', 'val.en']]

    with open(test_zh_path) as f_t_z, \
            open(test_en_path) as f_t_e:
         test_zh = f_t_z.readlines()
         test_en = f_t_e.readlines()

    '''    
    bs = 1
    if 'speed_baseline' not in sys.argv:
        bs = 16
        # 按zh长度粗排序
        test_zh, test_en = \
            zip(
                *(sorted(zip(test_zh, test_en),
                         key=lambda x: len(x[0]), reverse=True))
            )
    '''

    bs = 64
    start = 0
    pred_en = []
    test_en = test_en[100]
    for start in tqdm(range(0, len(test_en), bs)):
        en_sens = test_en[start:start + bs]
        zh_sens = test_zh[start:start + bs]
        pred_sens = predictor.predict(zh_sens)
        #print(f'org_en: \n {zh_sens}')
        #print(f'pred: \n {pred_sens}')
        #print(f'org_zh: \n {en_sens}')
        #sys.exit()
        pred_en.extend(pred_sens)
    print('pred_en len:',len(pred_en))
    print('test_en len:',len(test_en))

    pred_path_en = file_helper.get_abs_path(conf.eval_info.output)

    bleu(test_en, pred_en)

    with open(pred_path_en, 'w') as f_p:
        f_p.write('\n'.join(pred_en))
    

def bleu(gt, pred):
    gt = _split_texts(gt)
    pred = _split_texts(pred)
    #print(f"gt: {gt}")
    #print(f"pred: {pred}")
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
