#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from nmt_trans.tags.base_tag import BaseTagHelper, BaseTagEntity

class TagHelper(BaseTagHelper):
    '''
    用来支持对一些(百分号)数字规则的替换(直接复制)
    ｟n\d+｠
    '''
    num_pats = re.compile('('
        '[+-]?\d+\.?\d*\s?(?:%|pct|per cent|percent)' # -234.51 %
        '|\d{2}Q[1-4]' # Q4
        '|[1-4]Q\s?(?:FY)?\d{2,4}' # 1Q 2020
    ')')
    
    @property
    def label(self):
        return 'n'
    
    @property
    def en_pats(self):
        return TagHelper.num_pats
    
    @property
    def zh_pats(self):
        return TagHelper.num_pats
    
    @property
    def tag_entity_cls(self):
        return TagHelper._Num
    
    @property
    def upsample_ratio(self):
        return 2


    def _search(self, sen):
        matches = self.num_pats.finditer(sen)
        return [match.group(0) for match in matches]

    def tag_parallel(self, en_line, zh_line):
        # 英文百分号转义
        en_line = re.sub('(?<=\d)(\s?per\s?cent|\s?pct|\s?percent)', '%', en_line)
        return super().tag_parallel(en_line, zh_line)

    
    class _Num(BaseTagEntity):
        def __init__(self, num_str, lang):
            self.num_str = num_str
            self._hash = hash(num_str)
        
        def to_str(self, lang):
            return self.num_str
        
        def __str__(self):
            return self.to_str('en')
        
        def __eq__(self, other):
            return self.num_str == str(other)
        
        def __hash__(self):
            return self._hash


if __name__ == '__main__':
    tag_helper = TagHelper()
    test_en = 'Recent performance review: During the period from 26 June to 3 July, the banking sector rose by 6.75%, which was 0.03% lower than that of CSI 300 Index. Among them, state-owned banks rose by 4.36%, joint-stock banks rose by 7.72%, city commercial banks rose by 6.91%, and rural commercial banks rose by 11.27%.'
    test_zh = '近期表现回顾：06/26-07/03期间，银行板块涨幅6.75%，与沪深300相比跑输0.03个百分点。其中，国有银行涨幅4.36%，股份制银行涨幅7.72%，城商行涨幅6.91%，农商行涨幅11.27%。'
    new_en, en_id2dict = tag_helper.encode_en(test_en)
    new_zh, zh_id2dict = tag_helper.encode_zh(test_zh)
    print(new_en)
    print(new_zh)
    out_en = tag_helper.decode_en(new_en, en_id2dict)
    out_zh = tag_helper.decode_zh(new_zh, zh_id2dict)
    print(out_en)
    print(out_zh)
