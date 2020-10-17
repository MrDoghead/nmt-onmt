# nmt_trans
使用opennmt 来做翻译

##程序流程
### 设置环境变量
export PYTHONPATH=$PYTHONPATH:"PATH_TO/nmt_trans"
### 预处理数据
cd nmt_trans/preprocess

- 清洗翻译的语料： 去重, 并且进行分词，以及bpe
```
python prep_for_trans.py ../conf/chat_config.json
```
### 利用opennmt 训练模型
- 处理输入数据集， 构建词典
```
cd nmt_trans/preporcess
python precoss_for_nmt.py
```
- 训练

```
cd nmt_trans/train
CUDA_VISIBLE_DEVICES=0,2,4,5 nohup python train_nmt.py ../conf/chat_config.json > train_log2.txt &
```

- 模型转成高效的c++格式
```
cd nmt_trans/train
python convert_c_trans.py
```


### docker 打包部署

#### docker build
docker build -t forb_trans:v0.5.alpha . -f Dockerfile_cpu

#### 运行
docker run --name test_trans_nmt -d  -p 8884:8864 forb_trans:v0.5.alpha

#### 请求
curl --location --request POST 'http://103.28.213.215:8884/new_trans/en2zh' \
--header 'Content-Type: application/json' \
--data-raw '{"text_list": ["I love this game", "best practice"]} '