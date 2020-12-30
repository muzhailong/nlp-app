# nlp-app

**nlp-app主要实现了中文中主流的任务模型和前沿论文实现，如文本分类、标题生成、命名实体识别等等**

## 特点
- 使用简单，几乎不需要写其他代码就可以使用
- 使用horovod分布式训练，针对单机（多机）多卡训练更加快速
- 使用transformers，扩展性更好

## 进展
- [x] **文本分类**（类bert模型可直接使用）
- [x] **标题生成**（GPT2）
- [x] [**RealFormer**](https://arxiv.org/abs/2012.11747 "RealFormer")（GPT2上实现）
- [x] **根据关键词生成新闻**
----

## 文本分类
````python
./scripts/run_classification.sh
````
## 标题生成
#### 描述
使用微博数据集（大约45w条数据，[下载](https://www.jianshu.com/p/8f52352f0748?tdsourcetag=s_pcqq_aiomsg "下载")），根据文本生成title
样例：
```json
{
	"title": "人贩因抱孩子姿势不专业露馅",
	"content": "前天在郑州开往日照的2150次列车上，3岁孩子的母亲尹宁发现1名抱婴儿的女子根本不会照顾婴儿，基本不看孩子的脸。她怀疑是人贩，赶紧联系乘警。这名婴儿和与相关的5女1男被费县警方带走。昨天警方通报，2名嫌疑人涉嫌贩卖女婴被刑拘。给尹女士点[赞] "
}
```
#### 网络结构
config/title_generation_config.json
```json
{
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-4,
  "n_ctx": 512,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 6,
  "n_positions": 512,
  "vocab_size": 13317
}
```
可以直接修改title_generation_config.json文件来调整网络模型

#### 模型参数
| 参数  | 默认值  | 样例  | 解释  |
| ------------ | ------------ | ------------ | ------------ |
| batch_size  | 20  | 20  | 批量大小，注意这里的批量大小指的是每一张卡上的批量大小（和pytorch的数据并行不一样）  |
| base_lr | 4e-4  | 4e-4  | 学习率，此处学习率为单卡、无梯度累积时学习率  |
| max_len  | 512  | 512  | 序列最大长度，根据自己的机器和数据集进行调整  |
| do_train  | False  | False  | 是否训练  |
| do_predict  | False  | False  | 是否预测  |
| device  | cuda  | cuda  | 训练使用设备，默认使用cuda  |
| train_df_path  | 无  | /home/muzhailong/data/train.csv  | 训练使用的pandas DataFrame路径  |
| logdir  | ./log  | ./log  | 日志保存目录，可以用tensorboard可视化训练loss  |
| task_name  | 无  | generate_title  | 任务名称，随便起，主要用于在tensorboard中区分  |
| checkpoint_save_path  | 无  | /home/muzhailong/models/checkpoint.pth  | model optim epoch all_step 保存的路径，注意此代码把所有的信息保存到一个文件中  |
| base_model  | 无  | /home/muzhailong/nlp-app/config/config.json  | 此参数传递一个表示模型的json文件表示从头开始训练，传递一个transformers可以加载的模型文件路径将使用预训练模型  |
| tokenizer_model  | bert-base-chinese  | bert-base-chinese  |   |
| epochs  | 10  | 10  | 训练的批次   |
| save_steps  | 1000  | 1000  | 训练多少步保存一次  |
| warmup-epochs  | 2  | 2  | warup的epoch  |
| momentum  | 0.9  | 0.9  | 默认使用SGD   |
| wd  | 0.00005  | 0.00005  | weight decay  |
| compression_fp16  | False  | True  |分布式训练中数据压缩传输，在分布式环境下，建议带上   |
| batches-per-allreduce  | 1  |  3 | 梯度累积次数，根据自己的机器配置设定  |
| use-adasum | False | False | 梯度求和默认时求平均 |
| seed | 2020 | 2020 | 随机数种子 |
| max_grad_norm | 1.0 | 1.0 | 梯度裁剪，最大值 |


#### 数据拼接格式
```
[CLS] a1 a2 a3... [GENERATE_TITLE] b1 b2 b3 b4..
```
a1 a2 a3...:article token_id
b1 b2 b3...:标题token_id
\[GENERATE_TITLE\]:自己定义的一个token
注意该项目是在article token_id和[GENERATE_TITLE]之间填充到最大长度的

#### 使用
**训练**
```
1. 制作数据集，DF格式，其中包括两列，sentence1和lables
2. 根据自己机器配置修改模型参数（如：batch_size,epoch,分布式训练ip地址、保存地址等等）
3. 直接运行nlp-app/scripts/generate_title.sh
```
**生成**
```

```

## 根据关键词生成新闻
模型同title生成（仅仅数据拼接格式不同）
#### 使用
**训练**
```
1. 制作数据集，DF格式，其中包括两列，sentence1和labels，其中senetence1是关键词通过','.join(keywords)得到的，labels表示生成的新闻
2. 根据自己机器配置修改模型参数（如：batch_size,epoch,分布式训练ip地址、保存地址等等）
3. 直接运行nlp-app/scripts/keywords_to_news.sh
```
**生成**
```

```

## 引用与参考
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://github.com/google-research/bert "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding")
2. [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf "Language Models are Unsupervised Multitask Learners")
3. [transformers](https://github.com/huggingface/transformers "transformers")
4. [Morizeyao/GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese.git "Morizeyao/GPT2-Chinese")
5. [Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf "Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour")
6. [GPT2-NewsTitle](https://github.com/liucongg/GPT2-NewsTitle "GPT2-NewsTitle")
7. [定向写作模型CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION](https://arxiv.org/pdf/1909.05858.pdf "定向写作模型CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION")

> 本项目主要用来学习和竞赛使用，有错误的地方，请多多包含