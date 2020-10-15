#!/bin/bash

prefix="/data/translate/hb_trans"

toked_data="${prefix}/prep_corpus/prep_1kw_tagged_v3"
dest_dir="${prefix}/data_bin/nmt_1kw_v3"
mkdir -p ${dest_dir}
dest_data="${dest_dir}/tagged_en"
mode_dir="${prefix}/model_bin/nmt_1kw_v3"
mkdir -p ${mode_dir}
mode_path="${mode_dir}/mod_en"

onmt_preprocess \
	-train_src ${toked_data}/train.en -train_tgt ${toked_data}/train.zh \
	-valid_src ${toked_data}/val.en -valid_tgt ${toked_data}/val.zh \
	-save_data ${dest_data} -overwrite \
	-src_seq_length 200 -tgt_seq_length 200

# lbh 配置
python  train.py -data data/en2zh -save_model models/en2zh -world_size 6 -gpu_ranks 0 1 2 3 4 5 \
        -layers 4 -rnn_size 512 -word_vec_size 512 -batch_type tokens -batch_size 2048\
        -max_generator_batches 32 -normalization tokens -dropout 0.1 -accum_count 4 \
        -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer \
        -position_encoding -param_init 0 -warmup_steps 16000 -learning_rate 2 -param_init_glorot \
        -decay_method noam -label_smoothing 0.1 -adam_beta2 0.998 -report_every 1000


# faq 配置
CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" onmt_train -data ${dest_data} -save_model ${mode_path} \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 2048 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 16000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 5000 \
        -world_size 6 -gpu_ranks 0 1 2 3 4 5 

# translate 测试集
prefix="/data/translate/hb_trans"
dest_data="${prefix}/data_bin/all_v1_tagged_en"
mode_path="${prefix}/model_bin/all_v1_mod_en"
toked_data="${prefix}/prep_all_v1_tagged"
onmt_translate -gpu 7 -model ${mode_path}*50000.pt  -src ${toked_data}/val.en -tgt ${toked_data}/val.zh -replace_unk -verbose -output test_pred.tg.zh

#计算bleu分数
perl tools/multi-bleu.perl ${toked_data}/val.zh < test_pred.tg.zh

# 转换
ct2-opennmt-py-converter --model_path ${mode_dir}/mod_en_step_95000.pt --model_spec TransformerBase  --output_dir ${mode_dir}/c_trans_mod_en.pt
