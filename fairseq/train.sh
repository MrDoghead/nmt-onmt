
#NMT_PATH=$1
NMT_PATH=/data/tao.hu/nmt_trans
DATA_BIN=${NMT_PATH}/data/offline_data/fairseq_data_bin
SAVE_PATH=${NMT_PATH}/data/offline_data/model_bin/fairseq_checkpoints

rm -rf ${SAVE_PATH}
CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-train \
    ${DATA_BIN} \
    --save-dir ${SAVE_PATH} \
    --arch transformer_wmt_en_de \
    --task translation \
    --encoder-normalize-before --decoder-normalize-before \
    --max-source-positions 5000 --encoder-learned-pos \
    --max-target-positions 5000 --decoder-learned-pos \
    --left-pad-source False \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq 32 \
    --max-update 40000 \
    --keep-last-epochs 40 \
    --no-save-optimizer-state \
    --log-interval 10 \
    --fp16 \
    --log-format simple >& fairseq_train_log

