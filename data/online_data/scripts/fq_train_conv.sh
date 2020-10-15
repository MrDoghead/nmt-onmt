#!/bin/bash

preped_data="/data/translate/hb_trans/prep_corpus/prep_1kw_tagged_v2"
dest_data="/data/translate/hb_trans/data_bin/fq_1kw_en_zh_v2"
mkdir -p  ${dest_data}
SAVE="/data/translate/hb_trans/model_bin/fq_1kw_en_zh_conv_v2"
mkdir -p $SAVE

fairseq-preprocess --source-lang en --target-lang zh --trainpref ${preped_data}/train --validpref ${preped_data}/val --testpref ${preped_data}/val \
    --destdir ${dest_data}  --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

# for test
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=1,2,4,5 $(which fairseq-train) ${dest_data} \
-a fconv  --dropout 0.2 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer nag --clip-norm 0.1 \
--lr 0.5 --lr-scheduler fixed --force-anneal 50 \
--source-lang en --target-lang zh --max-tokens 4096  --log-interval 1000 --weight-decay 0.0001 \
--keep-last-epochs 10 \
--save-dir $SAVE \
--update-freq 4 \
--max-update 500000 \
--skip-invalid-size-inputs-valid-test


MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 $(which fairseq-train) ${dest_data} --clip-norm 0 --optimizer adam --lr 1e-3 --source-lang zh --target-lang en --max-tokens 3072  --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler inverse_sqrt --ddp-backend=no_c10d --max-update 500000 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 -a transformer --share-decoder-input-output-embed --save-dir $SAVE --dropout 0.3 --attention-dropout 0.1 --update-freq 4


python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

# 进行eval
# Evaluation
CUDA_VISIBLE_DEVICES=4 fairseq-generate $dest_data --path "${SAVE}/checkpoint_best.pt" --batch-size 128 --beam 8 --remove-bpe --lenpen 1 --gen-subset test --source-lang en --target-lang zh --quiet --skip-invalid-size-inputs-valid-test


CUDA_VISIBLE_DEVICES=4  fairseq-interactive $dest_data --path "${SAVE}/checkpoint_best.pt" --batch-size  64 --beam 8 --remove-bpe --source-lang en --target-lang zh --tokenizer moses --bpe subword_nmt --bpe-codes ${dest_data}/code

## 从服务器上截取下来的
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 $(which fairseq-train) ${dest_data} --clip-norm 0 --optimizer adam --lr 1e-3 --source-lang en --target-lang zh --max-tokens 4096  --log-interval 1000 --min-lr '1e-09' --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler inverse_sqrt --ddp-backend=no_c10d --max-update 500000 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 -a transformer --share-decoder-input-output-embed --save-dir $SAVE --dropout 0.3 --attention-dropout 0.1 --update-freq 4 --skip-invalid-size-inputs-valid-test
