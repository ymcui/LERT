[**中文**](https://github.com/ymcui/LERT) | [**English**](https://github.com/ymcui/LERT/blob/main/README_EN.md)

# LERT

LERT: A Linguistically-motivated Pre-trained Language Model (tentative)

## News

Something is on the way. Stay tuned and check back later.

**Tentative schedule:**

Update readme doc: late Oct, 2022

Release description paper: early Nov, 2022

Release pre-trained models (direct download): late Oct, 2022

Release pre-trained models (via huggingface): early Nov, 2022

Release pre-training scripts: late Nov - early Dec, 2022

----

[中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) |  [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner)

查看更多哈工大讯飞联合实验室（HFL）发布的资源：https://github.com/ymcui/HFL-Anthology

## 新闻
2022/10/18 感谢各位的关注，本项目在逐渐完善中。**内容不完整，相关信息待补充完善。**

## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [简介](#简介)                         | LERT预训练模型的基本原理                                       |
| [模型下载](#模型下载)         | LERT预训练模型的下载地址                               |
| [快速加载](#快速加载)                 | 如何使用[🤗Transformers](https://github.com/huggingface/transformers)快速加载模型 |
| [基线系统效果](#基线系统效果) | 中文NLU任务上的基线系统效果                             |
| [FAQ](#FAQ)                           | 常见问题答疑                                                 |
| [引用](#引用)                         | 本项目的技术报告                                          |


## 简介
TBA

## 模型下载

### TensorFlow 1.x版本（原版）

- 这里主要提供TensorFlow 1.15版本的模型权重。如需PyTorch或者TensorFlow 2版本的模型，请看下一小节。

- TensorFlow开源模型包含**完整权重**，包括MLM-head、linguistic-heads等。

| 模型简称                           | Layers | 隐层大小 |          注意力头          |          参数量        | Google下载 |                          百度盘下载                          |
| :--------------------------------- | :--: | :---------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **Chinese-LERT-large**        | 24 | 1024 |  16  |  330M  |   [TensorFlow]   | [TensorFlow（密码：）] |
| **Chinese-LERT-base**         | 12 | 768 |  12  |  110M   |   [TensorFlow]   | [TensorFlow（密码：）] |
| **Chinese-LERT-small** | 12 | 256 | 4 | 12M  |  [TensorFlow]   | [TensorFlow（密码：）] |

> *训练语料：中文维基百科，其他百科、新闻、问答等数据，总词数达5.4B，约占用20G磁盘空间，与MacBERT相同。  

以TensorFlow版`Chinese-LERT-base`为例，下载完毕后对zip文件进行解压得到：

```
chinese_lert_base_L-12_H-768_A-12.zip
    |- lert_model.ckpt      # 模型权重
    |- lert_model.meta      # 模型meta信息
    |- lert_model.index     # 模型index信息
    |- lert_config.json     # 模型参数
    |- vocab.txt            # 词表（与谷歌原版一致）
```

### PyTorch以及TensorFlow 2版本

- 通过🤗transformers模型库可以下载TensorFlow (v2)和PyTorch版本模型。
- PyTorch开源版本包含MLM部分的权重，但**不包含linguistic heads**。

下载方法：点击任意需要下载的模型 → 选择"Files and versions"选项卡 → 下载对应的模型文件。

| 模型简称 | 模型文件大小 | transformers模型库地址 |
| :------- | :---------: |  :---------- |
| **Chinese-LERT-large** | 1.2G | TBA |
| **Chinese-LERT-base** | ~400M | TBA |
| **Chinese-LERT-small** | ~60M | TBA |

## 快速加载
由于LERT主体部分仍然是BERT结构，用户可以使用[transformers库](https://github.com/huggingface/transformers)轻松调用LERT模型。

**注意：本目录中的所有模型均使用BertTokenizer以及BertModel加载。**

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```
其中`MODEL_NAME`对应列表如下：

| 模型名                 | MODEL_NAME                 |
| ---------------------- | -------------------------- |
| Chinese-LERT-large     | hfl/chinese-lert-large     |
| Chinese-LERT-base      | hfl/chinese-lert-base      |
| Chinese-LERT-small | hfl/chinese-lert-small |

## 基线系统效果
以下仅列举部分实验结果。详细结果和分析见论文。实验结果表格中，括号外为最大值，括号内为平均值。

### 中文任务

在以下10个任务上进行了效果测试。

- **抽取式阅读理解**（2）：[CMRC 2018（简体中文）](https://github.com/ymcui/cmrc2018)、[DRCD（繁体中文）](https://github.com/DRCKnowledgeTeam/DRCD)
- **文本分类**（6）：
  - **单句**（2）：[ChnSentiCorp](https://github.com/pengming617/bert_classification)、[TNEWS](https://github.com/CLUEbenchmark/CLUE)
  - **句对**（4）：[XNLI](https://github.com/google-research/bert/blob/master/multilingual.md)、[LCQMC](http://icrc.hitsz.edu.cn/info/1037/1146.htm)、[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)、[OCNLI](https://github.com/CLUEbenchmark/OCNLI)
- **命名实体识别**（2）：[MSRA-NER]()、[People's Daily（人民日报）]()

#### 阅读理解

TBA

#### 文本分类

TBA

#### 命名实体识别

TBA


## FAQ
TBA


## 引用
TBA


## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态。

![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)


## 问题反馈
如有问题，请在GitHub Issue中提交。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 重复以及与本项目无关的issue会被[stable-bot](stale · GitHub Marketplace)处理，敬请谅解。
- 我们会尽可能的解答你的问题，但无法保证你的问题一定会被解答。
- 礼貌地提出问题，构建和谐的讨论社区。
