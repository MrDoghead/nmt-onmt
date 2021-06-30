config='../conf/en_zh_config.json'
eval_data='../../data/offline_data/prep_corpus/prep_1kw_tagged_v2/'
echo "config file: ${config}"
echo "eval data: ${eval_data}"
python eval_en2zh.py ${config} ${eval_data}
