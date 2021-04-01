#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import numpy as np

class BaseTagHelper:
    def __init__(self):
        self.begin_sign = "｟"
        self.end_sign = "｠"
        self.label_pat = re.compile(rf'{self.begin_sign}({self.label}[\d]+){self.end_sign}')
        
        self.encode_en = lambda sen: self._encode(sen, 'en')
        self.encode_zh = lambda sen: self._encode(sen, 'zh')
        self.decode_en = lambda sen, id2entity: self._decode(sen, id2entity, 'en')
        self.decode_zh = lambda sen, id2entity: self._decode(sen, id2entity, 'zh')
    
    @property
    def label(self):
        raise NotImplementedError()
    
    @property
    def en_pats(self):
        raise NotImplementedError()
    
    @property
    def zh_pats(self):
        raise NotImplementedError()
    
    @property
    def tag_entity_cls(self):
        raise NotImplementedError()
    
    def _encode(self, sen, lang):
        pats = self.en_pats if lang == 'en' else self.zh_pats
        matches = pats.finditer(sen)
        words = [match.group(0) for match in matches]
        if len(words) < 1:
            return sen, {}
        words = set(words)
        id2entity = {}
        new_sen = sen
        for i, w in enumerate(words):
            new_sen = new_sen.replace(w, f'{self.begin_sign}{self.label}{i}{self.end_sign}')
            id2entity[self.label + str(i)] = self.tag_entity_cls(w, lang)
        return new_sen, id2entity
    
    def _decode(self, sen, id2entity, lang):
        '''
        @return
            str: 解码后的句子
            bool: 是否存在id不完全匹配
        '''
        if len(id2entity) < 1:
            return sen, True
        ids = set(id2entity.keys())
        ids_produced = set()
        new_sen = sen
        match_arr = self.label_pat.finditer(sen)
        for match in match_arr:
            match_str = match.group(0)
            id_ = match.group(1).strip()
            ids_produced.add(id_)
            entity = id2entity.get(id_, None)
            if entity is not None:
                new_sen = new_sen.replace(match_str, entity.to_str(lang))
        return new_sen, ids == ids_produced
    
    @property
    def cnt_start(self):
        '''
        tag标签起始位置，随机产生起始使得tag标签在语料中可以多样化
        '''
        return np.random.randint(50)
    
    @property
    def upsample_ratio(self):
        '''
        上采样数量，对于不同的tag应设置不同的上采样数量
        注意: 上采样可能会导致valid集数据泄露
        '''
        return 1
    
    def tag_parallel(self, en_line, zh_line):
        en_matches = self.en_pats.finditer(en_line)
        zh_matches = self.zh_pats.finditer(zh_line)
        en_dict = {}
        zh_dict = {}
        for match in en_matches:
            en_dict[self.tag_entity_cls(match.group(0), 'en')] = match.group(0)
        for match in zh_matches:
            zh_dict[self.tag_entity_cls(match.group(0), 'zh')] = match.group(0)
        en_tag_set = set(en_dict.keys())
        zh_tag_set = set(zh_dict.keys())
        tag_set = list(en_tag_set & zh_tag_set)
        if len(tag_set) == 0:
            return [(en_line, zh_line)]
        
        def repl(l, d, start):
            lid = start
            for p in tag_set:
                l = l.replace(d[p], f'{self.begin_sign}{self.label}{lid}{self.end_sign}')
                lid += 1
            return l
        cnt_starts = [self.cnt_start for _ in range(self.upsample_ratio)]
        return [(repl(en_line, en_dict, start),
                 repl(zh_line, zh_dict, start)) for start in cnt_starts]
    
class BaseTagEntity:
    def to_str(self, lang):
        raise NotImplementedError()

