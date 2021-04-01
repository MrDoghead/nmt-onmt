
#pip install fairseq==0.10.0
pip install pyarrow

#NMT_PATH=$1
NMT_PATH=/data/tao.hu/nmt_trans
DATA_BIN=${NMT_PATH}/data/offline_data/fairseq_data_bin


rm -rf ${DATA_BIN}
mkdir ${DATA_BIN}
fairseq-preprocess \
    --source-lang zh --target-lang en \
    --trainpref ${NMT_PATH}/data/offline_data/prep_corpus/prep_1kw_tagged_v2/train \
    --validpref ${NMT_PATH}/data/offline_data/prep_corpus/prep_1kw_tagged_v2/val \
    --testpref /data/tao.hu/haitong_test/test \
    --destdir ${DATA_BIN} \
    --workers 24


