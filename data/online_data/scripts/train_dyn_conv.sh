#!/bin/bash


#org_data="/data/translate/hb_trans/prep_ft_chinese"
#dest_data="/data/translate/hb_trans/data_bin/ft_chinese_en_zh"

org_data=$1
dest_data=$2


mkdir -p  ${dest_data}

fairseq-preprocess --source-lang en --target-lang zh --trainpref ${org_data}/train --validpref ${org_data}/val --testpref ${org_data}/val \
    --destdir ${dest_data}  --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

SAVE="DYCONV_MODEL_en_zh"
mkdir -p $SAVE

# 不加MKL_THREADING_LAYER=GNU 会报错，mkl 和numpy的
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=4,5,6,7 $(which fairseq-train) ${dest_data} --clip-norm 0 --optimizer adam --lr 0.0005 \
    --source-lang en --target-lang zh --max-tokens 4000  --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler inverse_sqrt --ddp-backend=no_c10d --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 \
    -a lightconv_wmt_zh_en_big --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 0 --decoder-glu 0 --no-progress-bar


# for larger_dataset
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup $(which fairseq-train) ${dest_data} --clip-norm 0 --optimizer adam --lr 0.0005 \
--source-lang en --target-lang zh --max-tokens 4000  --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler inverse_sqrt --ddp-backend=no_c10d --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 \
    -a lightconv_wmt_zh_en_big --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 0 --decoder-glu 0 --batch-size 256 --no-progress-bar  > log.txt &

# for test
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=4,5,6,7 nohup $(which fairseq-train) ${dest_data} --clip-norm 0 --optimizer adam --lr 0.0005 \
--source-lang en --target-lang zh --max-tokens 3072  --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler inverse_sqrt --ddp-backend=no_c10d --max-update 500000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 \
    -a lightconv_wmt_zh_en_big --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 0 --decoder-glu 0 --batch-size 256

python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

# 进行eval
# Evaluation
CUDA_VISIBLE_DEVICES=4 fairseq-generate $dest_data --path "${SAVE}/checkpoint_best.pt" --batch-size 128 --beam 8 --remove-bpe --lenpen 1 --gen-subset test --quiet


CUDA_VISIBLE_DEVICES=4  fairseq-interactive $dest_data --path "${SAVE}/checkpoint_best.pt" --batch-size  64 --beam 8 --remove-bpe --source-lang en --target-lang zh --tokenizer moses --bpe subword_nmt --bpe-codes ${dest_data}/code