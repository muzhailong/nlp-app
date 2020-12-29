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
- [ ] **命名实体识别**
----

## 文本分类
````python
./scripts/run_classification.sh
````
## 标题生成
```python
./models/generate_title.sh
```


## 引用与参考
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://github.com/google-research/bert "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding")
2. [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf "Language Models are Unsupervised Multitask Learners")
3. [transformers](https://github.com/huggingface/transformers "transformers")
4. [Morizeyao/GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese.git "Morizeyao/GPT2-Chinese")
5. [Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf "Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour")
6. [GPT2-NewsTitle](https://github.com/liucongg/GPT2-NewsTitle "GPT2-NewsTitle")

> 本项目主要用来学习和竞赛使用，有错误的地方，请多多包含