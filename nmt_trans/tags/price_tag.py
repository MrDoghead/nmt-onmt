#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
from nmt_trans.tags.base_tag import BaseTagHelper, BaseTagEntity

class TagHelper(BaseTagHelper):
    '''
    用来支持对一些货币金额规则的替换以及货币单位的转换
    ｟p\d+｠
    '''
    
    ''' 英文用前缀, 中文用后缀, list的第一个为翻译标准 '''
    units = [
        (["Rmb","RMB","rmb","CNY"], ["元","人民币"]),
        ([r"US\$","USD",r"US \$",r"\$"], ["美元","美金"]),
        (["EU","€"], ["欧元"]),
        (["GBP"], ["英镑"]),
        ([r"AU\$", "AUD",r"AU \$"], ["澳元","澳大利亚元"]),
        ([r"CA\$", "CAD",r"CA \$"], ["加元","加拿大元"]),
        (["JPY"], ["日元"]),
        ([r"HK\$","HKD",r"HK \$"], ["港元","港币"]),
#        (["IN"], ["卢比","印度卢比"]),
        ([r"NT\$","TWD",r"NT \$"], ["新台币"]),
        (["MOP"], ["澳门元"]),
        (["KRW"], ["韩元", "won"]),
        ([r"NZ\$","NZD",r"NZ \$"], ["新西兰元"]),
        (["BTC"], ["比特币","Bitcoin","bitcoin"]),
        (["THB"], ["泰铢"]),
        ([r"SG\$","SGD",r"SG \$"], ["新加坡元"]),
        (["ANG"], ["荷兰盾"]),
        (["CDF"], ["刚果法郎"]),
        (["CHF"], ["瑞士法郎"]),
        (["CLP"], ["智利比索"]),
        (["COP"], ["哥伦比亚比索"]),
        (["CUP"], ["古巴比索"]),
        (["CZK"], ["捷克克朗"]),
        (["SEK"], ["瑞典克朗"]),
        (["DEM"], ["德国马克"]),
        (["DKK"], ["丹麦克朗"]),
        (["FRF"], ["法郎","法国法郎"]),
        (["HUF"], ["匈牙利福林"]),
#        (["ID"], ["印度尼西亚卢比"]),
        (["ITL"], ["意大利里拉"]),
        (["MMK"], ["缅甸元"]),
        (["MXN"], ["墨西哥比索"]),
        (["NOK"], ["挪威克朗"]),
        (["PHP"], ["菲律宾比索"]),
        (["RUB"], ["卢布","俄罗斯卢布"]),
        (["VND"], ["越南盾"]),
    ]
    unit_2en = {}
    unit_2zh = {}
    # 去掉正则表达式里的转义符号
    _clear = lambda x: re.sub(r'\\', '', x)
    for en_l, zh_l in units:
        for i in range(len(zh_l)):
            unit_2en[_clear(zh_l[i])] = _clear(en_l[0])
            unit_2zh[_clear(zh_l[i])] = _clear(zh_l[0])
        for i in range(len(en_l)):
            unit_2zh[_clear(en_l[i])] = _clear(zh_l[0])
            unit_2en[_clear(en_l[i])] = _clear(en_l[0])

    number = r"[+-]?\s?\d+(?:,\d{3}){,5}(?:\.\d+)?" # 数字 +12.54 1,000,600
    amount_zh = r"[十百千万亿]*" # 量级
    amount_en = r"(billion|million|trillion|bn|mn|m|trn|k)?"
    curren_zh = '|'.join(['|'.join(zh_l) # 币种 新西兰元
                for en_l, zh_l in units])
    curren_en = '|'.join(['|'.join(en_l)
                for en_l, zh_l in units])
    splitter = r'[-/、和]|\s?and\s?'

    zh_price_pats = re.compile(
        rf'({number}\s?({splitter})\s?)''{,10}'
        rf'({number}\s?{amount_zh}\s?)'
        rf'({curren_zh})'
    )
    en_price_pats = re.compile(
        rf'({curren_en}\s?)'
        rf'({number}\s?{amount_en}\s?({splitter})\s?)''{,10}'
        rf'({number}\s?{amount_en})'
    )
    
    
    @property
    def label(self):
        return 'p'
    
    @property
    def en_pats(self):
        return TagHelper.en_price_pats
    
    @property
    def zh_pats(self):
        return TagHelper.zh_price_pats
    
    @property
    def tag_entity_cls(self):
        return TagHelper._Price
    
    @property
    def upsample_ratio(self):
        return 2


    zh_currency_p = re.compile(rf'^(.*?)({curren_zh})$')
    zh_amounts_p = re.compile(rf'(?<=\d)({amount_zh})')
    en_currency_p = re.compile(rf'^({curren_en})(.*?)$')
    en_amounts_p = re.compile(rf'(?<=\d)({amount_en})')
    spliter_p = re.compile(rf'.*?(?<=\d)({splitter})(?=[\d+-]).*')
    split_p = re.compile(rf'(?<=\d){splitter}(?=[\d+-])')
    class _Price(BaseTagEntity):
        '''
        中英不同货币格式的转换
        '''
        def __init__(self, price_str, lang):
            price_str = price_str.strip()
            self.currency, self.spliter, self.num = self._from_en(price_str) if lang == 'en' else \
                                                    self._from_zh(price_str)
            self.currency_en, self.amount_en, self.num_en = self._to('en')
            self.currency_zh, self.amount_zh, self.num_zh = self._to('zh')
            self._hash = hash(self.currency_en + self.amount_en + ''.join(self.num_en))
        
        trans_amount_zh = lambda x: 10**(x.count('十') * 1 +
                                         x.count('百') * 2 +
                                         x.count('千') * 3 +
                                         x.count('万') * 4 +
                                         x.count('亿') * 8)
        def _from_zh(self, price):
            # 1. 提取货币单位currency
            m = TagHelper.zh_currency_p.fullmatch(price)
            others = m.group(1).replace(' ', '').replace(',', '')
            currency = m.group(2)
            currency = TagHelper.unit_2en[currency]
            # 2. 提取量级amounts
            amounts = []
            for m in TagHelper.zh_amounts_p.finditer(others):
                if len(m.group(0)) > 0:
                    amounts.append(m.group(0))
            others = TagHelper.zh_amounts_p.sub('', others)
            # 3. 取得spliter并对可能对多个金额数字做切割,
            # 可能会有多个spliter (e.g. 1.5、2.2和3), 取第一个为准
            spliter = ''
            m = TagHelper.spliter_p.match(others)
            if m:
                spliter = m.group(1).strip()
                num = TagHelper.split_p.split(others)
            else:
                num = [others]
            # 4. 计算数额
            num = np.array([float(n) for n in num], dtype=np.float64)
            if len(amounts) == 0:
                amounts = ['']
            if len(amounts) != len(num):
                amounts = [amounts[-1]]
            amounts = [TagHelper._Price.trans_amount_zh(amount) for amount in amounts]
            num *= amounts
            return currency, spliter, num

        trans_amount_en = {'k':1e3, 'm':1e6, 'mn':1e6, 'million':1e6,
                           'bn':1e9, 'billion':1e9, 'trn':1e12,
                           'trillion':1e12, '':1}
        def _from_en(self, price):
            m = TagHelper.en_currency_p.fullmatch(price)
            others = m.group(2).replace(' ', '').replace(',', '')
            currency = m.group(1)
            currency = TagHelper.unit_2en[currency]
            amounts = []
            for m in TagHelper.en_amounts_p.finditer(others):
                if len(m.group(0)) > 0:
                    amounts.append(m.group(0))
            others = TagHelper.en_amounts_p.sub('', others)
            spliter = ''
            m = TagHelper.spliter_p.match(others)
            if m:
                spliter = m.group(1).strip()
                num = TagHelper.split_p.split(others)
            else:
                num = [others]
            num = np.array([float(n) for n in num], dtype=np.float64)
            if len(amounts) == 0:
                amounts = ['']
            if len(amounts) != len(num):
                amounts = [amounts[-1]]
            amounts = [TagHelper._Price.trans_amount_en[amount] for amount in amounts]
            num *= amounts
            return currency, spliter, num

        def _to(self, lang):
            currency = self.currency if lang == 'en' else \
                       TagHelper.unit_2zh[self.currency]
            amount = ''
            num = np.array(self.num)
            thres_amounts = zip([1e12, 1e9, 1e6, 1e3], ['trn', 'bn', 'mn', 'k']) if lang == 'en' else \
                            zip([1e12, 1e8, 1e4], ['万亿', '亿', '万'])
            
            for threshold, amount_name in thres_amounts:
                if np.max(self.num) >= threshold or np.min(self.num) <= -threshold:
                    amount = amount_name
                    num /= threshold
                    break
            num = np.around(num, decimals=6)
            
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
            
            return currency, amount, [num2str(n) for n in num]
            
        
        def to_str(self, lang):
            if lang == 'en':
                return ' {}{}{} '.format(
                    self.currency_en,
                    self.spliter.join(self.num_en),
                    self.amount_en
                )
            else:
                return '{}{}{}'.format(
                    self.spliter.join(self.num_zh),
                    self.amount_zh,
                    self.currency_zh
            )
        
        def __str__(self):
            return self.to_str('en')
        
        def __eq__(self, other):
            if not isinstance(other, TagHelper._Price):
                return False
            return self.currency_en == other.currency_en and \
                   self.amount_en == other.amount_en and \
                   self.num_en == other.num_en
                   
        def __hash__(self):
            return self._hash

if __name__ == '__main__':
    tag_helper = TagHelper()
    test = [
        ("企业贷款增加5797亿元，少增716亿元，其中主要是票据融资减少了4102亿元，但中长期贷款和短贷分别同比上升了2967亿元和402亿元，整个企业的经济活动明显复苏。",
         "Corporate loans increased by Rmb579.7bn, -Rmb71.6bn YoY;the main reason is that bill financing decreased by Rmb410.2bn, but mid-to-long term loans and short-term loans increased by Rmb296.7bn and Rmb40.2bn YoY, respectively.The corporate sector’s economic activity has obviously recovered."),
        ("归母净利润规模基本维持在 550-700 亿元（2011 年最高达 702.55 亿元）；",
         "and the net profit attributable to shareholders (NPAtS) remained at Rmb55-70bn (up to Rmb70.255bn in 2011)"),
        ("我们预测公司2020-2022年EPS分别为0.42、2.35、3.62元/股。",
         "We predict that the EPS of the company in 2020-2022 will be Rmb0.42/2.35/3.62/share respectively."),
        ("EPS由0.73/0.74/0.75元上调为1.02/ 0.98/ 0.91元，BVPS由9.76/10.27/10.79元上调为9.96/ 10.64/ 11.25元。",
         "We raise the company's 2020-2022E financial forecast EPS from Rmb0.73/-0.74/0.75 to Rmb1.02/ 0.98and 0.91, BVPS from Rmb9.76/10.27/10.79 to Rmb9.96/10.64/11.25."),
        ("8月份前9天的日均博彩总收入为3300万澳门元，低于7月份的4300万澳门元。",
         "GGR per day in the first nine days of Aug was MOP33mn, lower than that of July at MOP43mn."),
        ("预计2020-2022年收入各1,006亿美元、1,055亿元、1,141亿元；归母净利润32亿元、35.7亿元、41.2亿元。对应EPS分别0.34、0.37和0.43元。",
         "We estimate 2020-2022 revenue to be $100.6bn, Rmb105.5bn and Rmb114.1bn respectively, NP to be Rmb3.2bn, Rmb3.57bn and Rmb4.12bn, and corresponding EPS to be Rmb0.34, Rmb0.37 and Rmb0.43 respectively."),
        ("A/60/4440-S/2005/658 项目I9、 12、 14、 15、 17、 19、 24、 25、 27、 30、 31、 32、 33、 38、 39、 40、 41、 42、 43、 46、 50、 51、 52、 54、 56、 66、 69、 70、 71、 73、 74、 84、 89、 90、 94、 97、 100、 103、 108、 110、 115、 116、 117、 118、 119 和 120 -- -- 2005年10月17日也门常驻联合国代表给秘书长的信 [阿、中、英、法、俄、西]",
         "USD2000--2500,$353---223, Rmb55bn-70bn, Rmb-55m -70bn, Rmb5/ 24 m / 56763k/642,311bn/432,2mn"),
        ('，发卡量分别为183.29万张、155.41万张、289.68万张和188.28万张；公司信用卡科技服务分别实现收入4,857.58万元、6,289.95万元、11,945.87万元和12,499.85万元',
         'the number of cards issued was 1,832,900, 1,554,100, 2,896,800 and 1,882,800 respectively; The company s credit card technology services achieved revenue of Rmb48,575,800, Rmb62,899,500, Rmb119,458,700 and Rmb124,998,500 respectively.')
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
