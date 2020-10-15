#!/usr/bin/bash

init_data="/data/translate/hb_trans/raw_para_corpus/all_1kw"
org_data="/data/translate/hb_trans/raw_para_corpus/all_1kw_tagged_v3"
preped_data="/data/translate/hb_trans/prep_corpus/prep_1kw_tagged_v3"
f_base_name="all"

# 主要用来做替换
python pre_process/offline/add_tag.py --input ${init_data} --output ${org_data} --b_name ${f_base_name}

# 主要是用来做分词，以及预处理；
python pre_process/prepare_nmt_parallel.py --input ${org_data} --output ${preped_data} --fbase ${f_base_name}
