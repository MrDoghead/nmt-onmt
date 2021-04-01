#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from nmt_trans.tags.base_tag import BaseTagHelper, BaseTagEntity

class TagHelper(BaseTagHelper):
    '''
    用来支持一些时间(具体日期+财年季度)的替换
    ｟d\d+｠
    '''
    
    '''
    # TODO quater 移到这里
    17年4季度
    4Q17 2019Q1
    4季度
    Q4
    '''
    
    ''' list第一个为翻译导出标准 '''
    _months = [
        (['1', '01'], ['Jan', 'January', '1', '01']),
        (['2', '02'], ['Feb', 'February', '2', '02']),
        (['3', '03'], ['Mar', 'March', '3', '03']),
        (['4', '04'], ['Apr', 'April', '4', '04']),
        (['5', '05'], ['May', 'May', '5', '05']),
        (['6', '06'], ['Jun', 'June', '6', '06']),
        (['7', '07'], ['Jul', 'July', '7', '07']),
        (['8', '08'], ['Aug', 'August', '8', '08']),
        (['9', '09'], ['Sep', 'September', '9', '09']),
        (['10'], ['Oct', 'October', '10']),
        (['11'], ['Nov', 'November', '11']),
        (['12'], ['Dec', 'December', '12'])
    ]
    month_2en = {}
    month_2zh = {}
    for zh_m, en_m in _months:
        for i in range(len(en_m)):
            month_2en[en_m[i]] = en_m[0]
            month_2zh[en_m[i]] = zh_m[0]
        for i in range(len(zh_m)):
            month_2en[zh_m[i]] = en_m[0]
            month_2zh[zh_m[i]] = zh_m[0]

    day_num = '([0-2][0-9]|3[01]|[1-9])'
    month_num = '(' + '|'.join('|'.join(zh_m) for zh_m, en_m in _months) + ')'
    day_zh = day_num
    month_zh = month_num
    day_en = (
        r'('
        r'[0-3]?1(st)?|'
        r'[0-2]?2(nd)?|'
        r'[0-2]?3(rd)?|'
        r'[0-2]?[04-9](th)?|'
        r'30|30th'
        r')'
    )
    month_en = '(' + '|'.join('|'.join(en_m) for zh_m, en_m in _months) + ')'
    year = '((19|20)[0-9][0-9])' # 暂时只支持4位数年份
    
    spliter = '/'
    
    zh_date_pats = re.compile(
         r'(?<![到至\-0-9])'
         r'('
        rf'(({year}\s?年\s?)?{month_zh}\s?月份?\s?{day_num}[日号])|'         # 2019年5月1日, 5月1日
        rf'({year}\s?年\s?{month_zh}月份?)|'                                # 2019年5月
        rf'(({year}\s?{spliter}\s?)?{month_zh}\s?{spliter}\s?{day_num})|'  # 2019/5/1, 5/1
        rf'({year}\s?{spliter}\s?{month_zh})'                              # 2019/5
         r')'
         r'(?![到至\-]| -)'                                                 # 暂时不支持范围
    )
    
    # TODO 注意 en_date_pats的匹配覆盖率不高，目前不适合英翻中情景 
    en_date_pats = re.compile(
         r'(?<![0-9a-zA-Z-])'
         r'('
        rf'(({year}\s?{spliter}\s?)?{month_num}\s?{spliter}\s?{day_en})|'  # 2019/5/1, 5/1
        rf'({year}\s?{spliter}\s?{month_num})|'                            # 2019/5
        rf'(({day_en}\s)?{month_en}\s{year})|'                              #  1st Dec 2015, Dec 2015
        rf'({day_en}\s{month_en})|({month_en}\s{day_en})'                 # 5 Dec, Dec 15, Dec 15th
         r')'
         r'(?![0-9a-zA-Z-])'
    )
    
    quater_pats = ''
    
    
    @property
    def label(self):
        return 'd'
    
    @property
    def en_pats(self):
        return TagHelper.en_date_pats
    
    @property
    def zh_pats(self):
        return TagHelper.zh_date_pats
    
    @property
    def tag_entity_cls(self):
        ''' tag双语部分只需要处理日期时间，财年季度的处理不必要 '''
        return TagHelper._Date
    
    def _encode(self, sen, lang):
        date_pats = self.en_pats if lang == 'en' else self.zh_pats
        matches = date_pats.finditer(sen)
        words = [match.group(0) for match in matches]
        if len(words) < 1:
            return sen, {}
        words = set(words)
        id2date = {}
        new_sen = sen
        for i, w in enumerate(words):
            new_sen = new_sen.replace(w, f'{self.begin_sign}{self.label}{i}{self.end_sign}')
            id2date['d' + str(i)] = self.tag_entity_cls(w, lang)
        # TODO quater (直接复制)
        return new_sen, id2date
    
    @property
    def upsample_ratio(self):
        return 1
    
    from_zh_p = re.compile(
        r'^'
        r'((?P<year>\d{4})\s?年\s?)?'
        r'(?P<month>\d{1,2})\s?月份?\s?'
        r'((?P<day>\d{1,2})\s?[日号]\s?)?'
        r'$'
    )
    from_en_p = re.compile(
         r'^'
        rf'((?P<day_1>{day_en})\s)?(?P<month_1>{month_en})(\s(?P<year_1>{year}))|'
        rf'(?P<day_2>{day_en})\s(?P<month_2>{month_en})|'
        rf'(?P<month_3>{month_en})\s(?P<day_3>{day_en})'
         r'$'
    )
    
    class _Date(BaseTagEntity):
        def __init__(self, date_str, lang):
            date_str = date_str.strip()
            self.year, self.month, self.day, self.spliter = \
                self._from_zh(date_str) if lang == 'zh' else \
                self._from_en(date_str)
            self.year_en, self.month_en, self.day_en = self._to('en')
            self.year_zh, self.month_zh, self.day_zh = self._to('zh')
            self._hash = hash('{}y{}m{}d'.format(self.year, self.month, self.day))
            
        def _from_zh(self, date_str):
            if TagHelper.spliter in date_str:
                return self._from_spliter(date_str)
            match = TagHelper.from_zh_p.match(date_str)
            y, m, d = match.group('year'), match.group('month'), match.group('day')
            y = int(y) if y else None
            m = int(m) if m else None
            d = int(d) if d else None
            return y, m, d, False

        def _from_en(self, date_str):
            if TagHelper.spliter in date_str:
                return self._from_spliter(date_str)
            match = TagHelper.from_en_p.match(date_str)
            y = match.group('year_1')
            m = match.group('month_1') or match.group('month_2') or match.group('month_3')
            d = match.group('day_1') or match.group('day_2') or match.group('day_3')
            y = int(y) if y else None
            m = int(TagHelper.month_2zh[m]) if m else None
            if d:
                if d[-2:] in ['st', 'nd', 'rd', 'th']:
                    d = d[:-2]
                d = int(d)
            return y, m, d, False
            

        def _from_spliter(self, date_str):
            try: # TODO 正则
                date_str = date_str.replace(' ', '')
                y = m = d = None
                temp = date_str.split(TagHelper.spliter)
                if len(temp) == 3:
                    y, m, d = temp
                    if d and d[-2:] in ['st', 'nd', 'rd', 'th']:
                        d = d[:-2]
                    y, m, d = int(y), int(m), int(d)
                else:
                    if len(temp[0]) == 4:
                        y, m = temp
                        y, m = int(y), int(m)
                    else:
                        m, d = temp
                        if d and d[-2:] in ['st', 'nd', 'rd', 'th']:
                            d = d[:-2]
                        m, d = int(m), int(d)
                return y, m, d, True
            except:
                return None, None, None, True
        
        def _to(self, lang):
            if lang == 'en':
                year_en = str(self.year) if self.year else self.year
                month_en = TagHelper.month_2en[str(self.month)] if self.month else \
                           self.month
                day_en = str(self.day) if self.day else self.day
                return year_en, month_en, day_en
            else:
                year_zh = str(self.year) + '年' if self.year else self.year
                month_zh = TagHelper.month_2zh[str(self.month)] + '月' if self.month else \
                           self.month
                day_zh = str(self.day) + '日' if self.day else self.day
                return year_zh, month_zh, day_zh

        def to_str(self, lang):
            if self.spliter:
                ''' 如果是2010/11/5的带有spliter'/'的形式则直接复制 '''
                temp = []
                for x in [self.year, self.month, self.day]:
                    if x:
                        temp.append(str(x))
                return TagHelper.spliter.join(temp)
            
            if lang == 'en':
                temp = []
                for x in [self.day_en, self.month_en, self.year_en]:
                    if x:
                        temp.append(x)
                return ' '.join(temp)
            else:
                temp = []
                for x in [self.year_zh, self.month_zh, self.day_zh]:
                    if x:
                        temp.append(x)
                return ''.join(temp)

        def __str__(self):
            return self.to_str('en')

        def __eq__(self, other):
            if not isinstance(other, TagHelper._Date):
                return False
            return self._hash == other._hash and \
                   self.year == other.year and \
                   self.month == other.month and \
                   self.day == other.day
        
        def __hash__(self):
            return self._hash


        
if __name__ == '__main__':
    tag_helper = TagHelper()
    test = [
        ("2000/1 2011/5 2011/4/1, 5 December， 14 Dec 2011， 03rd July 2018， 05/6-07/10, 2015 Apr, Nov 1990.",
         ",2011/4/1，5月3日，6月1号到5日 5月1号至8月1日，2018年7月03号，05/6-07/10, 15年4月"),
        ("On September 30, 2019, the prices, exports. On October 25, 2018, the, from November 1, 2018.",
         "， 2019年9月30日三者,力。2018年10月25日财政部发布的《关于调整部分产品出口退税率的通知》，自2018年11月1日起，"),
        ("bcasting from 24th Feb to 1 Mar, 2020,  at 28x FY20 PE on 26 Feb 2020.",
         "2020.2.24-3.1电, 15%，目前（20200226）收盘 "),
        ("and land management (E/CN.16/1997/8)  8/27th 8/28th",
         "合说明(E/CN.16/1997 /8)"),
        ("20201225-20210101",
         "20201225-20210101"),
    ]
    for test_en, test_zh in test:
        new_en, en_id2dict = tag_helper.encode_en(test_en)
        new_zh, zh_id2dict = tag_helper.encode_zh(test_zh)
        print(new_en)
        print(new_zh)
        out_en = tag_helper.decode_en(new_en, en_id2dict)
        out_zh = tag_helper.decode_zh(new_zh, zh_id2dict)
        print(out_en)
        print(out_zh)
        print('\n')
    # tag_parallel
    for test_en, test_zh in test:
        en, zh = tag_helper.tag_parallel(test_en, test_zh)[0]
        print(en)
        print(zh)