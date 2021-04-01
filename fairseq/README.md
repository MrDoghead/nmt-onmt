
### Torch
```bash
pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

### Fairseq
```bash
git clone git@git.aigauss.com:tao.hu/fairseq.git
cd fairseq
pip install --editable ./
```

### Train
```bash
sh preprocess.sh
sh train.sh
python fairseq/scripts/average_checkpoints.py \
    --inputs [CKPT_DIR] --output [OUT_PATH] \
    --num-epoch-checkpoints [NUM] --checkpoint-upper-bound [BOUND]
```

### Transfer
```bash
python fairseq2onmt.py [Fairseq_model_file] [Target_model_file] [ |large]
```

