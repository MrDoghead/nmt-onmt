# nmt_trans
使用opennmt 来做翻译

##程序流程
### 设置环境变量
export PYTHONPATH=$PYTHONPATH:"PATH_TO/chat_bot"
### 预处理数据
cd hb_chat/preporcess
- 清洗翻译的语料： 去重等
```
python process_org_data.py
```
- 将语料准备成平行语料， 并且进行分词，以及bpe
```
python data_prep.py
```
### 利用opennmt 训练模型
- 处理输入数据集， 构建词典
```
cd hb_chat/preporcess
python precoss_for_nmt.py
```
- 训练

```
cd hb_chat/train
python train.py
```
- 评估
```
python evaluate.py
```
- 模型转成高效的c++格式
```
python convert_c_trans.py
```
