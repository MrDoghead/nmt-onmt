# nmt_trans
本项目主要基于opennmt 和 fairseq 来做翻译

##预处理流程
### 设置环境变量
export PYTHONPATH=$PYTHONPATH:"PATH_TO/nmt_trans"
### 预处理数据
cd nmt_trans/preprocess 

- 清洗翻译的语料： 去重, 并且进行分词，以及bpe
```
python prep_for_trans.py ../conf/en_zh_config.json
```

## opennmt pipeline
### 利用opennmt 训练模型
- 处理输入数据集， 构建词典
```
cd nmt_trans/preporcess
python precoss_for_nmt.py ../conf/en_zh_config.json
```

- 训练

```
cd nmt_trans/train
nohup python train_nmt.py ../conf/en_zh_config.json > train_log.txt &
```

- average ckpts
```
onmt_average_models -models MODELs -output OUTPUT
```

- 模型转成高效的c++格式
```
cd nmt_trans/train
python convert_c_trans.py ../conf/en_zh_config.json
```
- 模型评估
```
cd nmt_trans/train
python eval_en2zh.py ../conf/en_zh_config.json /path/to/test_data
```

## Fairseq pipeline
- 处理输入数据集， 构建词典
```
cd nmt_trans/preporcess
python preprocess_4_fairseq.py ../conf/fairseq_train_en_zh_lg_laydrop.json
```

- 训练

```
cd nmt_trans/train
nohup python train_fairseq.py ../conf/fairseq_train_en_zh_lg_laydrop.json > train_log.txt &
```

- average ckpts
```
python /path_to/fairseq/scripts/average_checkpoints.py \
    --inputs [CKPT_DIR] --output [OUT_PATH] \
    --num-epoch-checkpoints [NUM] 
```

- 模型评估
```
cd nmt_trans/train
python eval_en2zh_fq.py ../conf/fairseq_train_en_zh_lg_laydrop.json /path/to/test_data
```


## docker 打包部署

#### docker build
docker build -t forb_trans:v0.5.alpha . -f Dockerfile_cpu

#### 运行
docker run --name test_trans_nmt -d  -p 8884:8864 forb_trans:v0.5.alpha

#### 请求
curl --location --request POST 'http://ip:port/new_trans/en2zh' \
--header 'Content-Type: application/json' \
--data-raw '{"text_list": ["I love this game", "best practice"]} '