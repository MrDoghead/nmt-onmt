#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
from nmt_trans.tags.base_tag import BaseTagHelper, BaseTagEntity

class TagHelper(BaseTagHelper):
    '''
    用于支持一些单位的转换 e.g. 1万->10k
    ｟a\d+｠
    需要在price_tag之后处理
    '''
    
    num = r"\d+(?:,\d{3}){,5}(?:\.\d+)?"
    
    zh_amount_pats = re.compile(
        rf'{num}'
         r'[十百千万亿]{1,5}'
    )
    en_amount_pats = re.compile(
        rf'{num}\s?'
         r'(billion|million|thousand|trillion|bn|mn|m|trn|k)?'
         r'(?![a-z])'
    )
    
    @property
    def label(self):
        return 'a'
    
    @property
    def en_pats(self):
        return TagHelper.en_amount_pats
    
    @property
    def zh_pats(self):
        return TagHelper.zh_amount_pats
    
    @property
    def tag_entity_cls(self):
        return TagHelper._Amount

    @property
    def upsample_ratio(self):
        return 8
    
    zh_amount_p = re.compile(
        rf'^(?P<n1>{num})(?P<n2>[十百千万亿]''{1,5})$'
    )
    en_amount_p = re.compile(
         r'^'
        rf'(?P<n1>{num})\s?'
         r'(?P<n2>billion|million|thousand|trillion|bn|mn|m|trn|k)?'
         r'$'
    )
    class _Amount(BaseTagEntity):
        def __init__(self, amount_str, lang):
            self.amount = self._from_en(amount_str) if lang == 'en' else \
                          self._from_zh(amount_str)
            self.amount_en = self._to('en')
            self.amount_zh = self._to('zh')
            self._hash = hash(str(self.amount))

        trans_amount_zh = lambda x: 10**(x.count('十') * 1 +
                                         x.count('百') * 2 +
                                         x.count('千') * 3 +
                                         x.count('万') * 4 +
                                         x.count('亿') * 8)
        def _from_zh(self, amount):
            m = TagHelper.zh_amount_p.fullmatch(amount)
            n1, n2 = m.group(1), m.group(2)
            n1 = n1.replace(',', '')
            return np.array([float(n1)], dtype=np.float64) * \
                   TagHelper._Amount.trans_amount_zh(n2)

        trans_amount_en = {'k':1e3, 'm':1e6, 'mn':1e6, 'million':1e6,
                           'bn':1e9, 'billion':1e9, 'trn':1e12,
                           'trillion':1e12, 'thousand':1e3, '':1, None:1}            
        def _from_en(self, amount):
            m = TagHelper.en_amount_p.fullmatch(amount)
            n1, n2 = m.group(1), m.group(2)
            n1 = n1.replace(',', '')
            return np.array([float(n1)], dtype=np.float64) * \
                   TagHelper._Amount.trans_amount_en[n2]
            
        def _to(self, lang):
            thres_amounts = zip([1e12, 1e9, 1e6], ['trillion', 'billion', 'million']) if lang == 'en' else \
                            zip([1e12, 1e8, 1e4], ['万亿', '亿', '万'])
            amount_unit = ''
            amount_num = self.amount
            for threshold, amount_name in thres_amounts:
                if self.amount >= threshold:
                    amount_unit = amount_name
                    amount_num = self.amount / threshold
                    break
            amount_num = np.around(np.array(amount_num, dtype=np.float64), decimals=6)
            amount_num = amount_num[0]
            
            def num2str(x):
                if x < 0:
                    return '-' + num2str(-x)
                x = str(int(x)) if x == int(x) else str(x)
                # 每三个数字加个逗号
                x = x.split('.')
                integer = x[0]
                temp = []
                l, k = len(integer), (len(integer) - 1) // 3
                for i in range(k):
                    temp.append(integer[l-3-3*i:l-3*i])
                temp.append(integer[:l-3*k])
                x[0] = ','.join(reversed(temp))
                return '.'.join(x)
            
            return num2str(amount_num) + amount_unit if lang == 'zh' else \
                   num2str(amount_num) + ' ' + amount_unit

        def to_str(self, lang):
            return self.amount_en if lang == 'en' else \
                   self.amount_zh
        
        def __str__(self):
            return self.to_str('en')

        def __eq__(self, other):
            if not isinstance(other, TagHelper._Amount):
                return False
            return self._hash == hash(other)

        def __hash__(self):
            return self._hash

if __name__ == '__main__':
    tag_helper = TagHelper()
    test = [
        ('港奶粉数量为2.21万吨，环比-17.0%，同比-9.1%；进口到港鲜奶数量为1.77万吨',
         'Hong Kong was 22,100 tons, which was -17.0% MoM and -9.1% YoY; the quantity of fresh milk imported to Hong Kong was 17,700 tons'),
        ('牛奶产量有望达到96亿升，比当前预期产量多了10亿升',
         ' milk production is expected to reach 9.6bn liters, 1bn liters more than the current expected output'),
        ('，发卡量分别为183.29万张、155.41万张、289.68万张和188.28万张；公司信用卡科技服务分别实现收入4,857.58万元、6,289.95万元、11,945.87万元和12,499.85万元',
         'the number of cards issued was 1,832,900, 1,554,100, 2,896,800 and 1,882,800 respectively; The company s credit card technology services achieved revenue of Rmb48,575,800, Rmb62,899,500, Rmb119,458,700 and Rmb124,998,500 respectively.'),
        ('达31150亿元',
         'reached Rmb311.5bn'),
        ('销量0.2亿件',
         'was 20mn')
    ]
    # encode / decode
    for test_zh, test_en in test:
        new_en, en_id2price = tag_helper.encode_en(test_en)
        new_zh, zh_id2price = tag_helper.encode_zh(test_zh)
        print(new_en)
        print(new_zh)
        out_en = tag_helper.decode_zh(new_en, en_id2price)
        out_zh = tag_helper.decode_en(new_zh, zh_id2price)
        print(out_en)
        print(out_zh)
    print('\n')
    # tag_parallel
    for test_zh, test_en in test:
        en, zh = tag_helper.tag_parallel(test_en, test_zh)[0]
        print(en)
        print(zh)
    
