[**中文**](https://github.com/ymcui/LERT) | [**English**](https://github.com/ymcui/LERT/blob/main/README_EN.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="400"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/ymcui/LERT/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/LERT.svg?color=blue&style=flat-square">
    </a>
</p>


It is generally believed that Pre-trained Language Model (PLM) has the ability to automatically learn linguistic knowledge from large-scale text corpora. In order to verify whether the pre-trained language model can be further improved by explicitly injecting linguistic knowledge, in this project, the Joint Laboratory of HIT and iFLYTEK Research  (HFL) proposed <b>a new pre-trained model LERT</b> with enhanced linguistic information, which incorporates a variety of linguistic knowledge. Extensive experimental results show that LERT can bring significant performance improvement under the same training data scale. LERT-related resources will be open-sourced for promoting academic research.


- Paper: TBA (expected November 2022)

----

[Chinese and English PERT](https://github.com/ymcui/PERT) | [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [ Knowledge distillation tool TextBrewer](https://github.com/airaria/TextBrewer) | [Model cropping tool TextPruner](https://github.com/airaria/TextPruner)

View more resources released by HFL: https://github.com/ymcui/HFL-Anthology

## News
2022/10/26 **Model download links and baseline system results have been updated**. The rest of the information will be added later.

2022/10/18 Thank you for your attention, this project is gradually improving the content.

## Table of Contents
| Chapter | Description |
| ------------------------------------- | ----------- ---------------------------- --------------------- |
| [Introduction](#Introduction) | Introduction of LERT |
| [Download](#Download) | Download links for LERT |
| [QuickLoad](#QuickLoad) | How to use [ 🤗 Transformers](https://github.com/huggingface/transformers) to quickly load models|
| [Baselines](#Baselines) | Baseline system results on Chinese NLU tasks |
| [FAQ](#FAQ) | Frequently Asked Questions|
| [Citation](#Citation) | Technical report of this project|


## Introduction
TBA

## Model download

### TensorFlow 1.x version (original)

- The model weights of the TensorFlow 1.15 version are mainly provided here. See the next section to get models for PyTorch or TensorFlow 2.
- The open-sourced TensorFlow models include **full weights**, including MLM-head, linguistic-heads, etc.

| Model | Layers | Hidden size | Attention head | Params | Google Drive | Baidu Disk |
| :---------------------------------- | :--: | :-------- -------------: | :------------: | :------------- -----------------------------------: | :------------ --------------------------------------------------------: | :- -------------------------------------------------- -------: |
| **Chinese-LERT-large** | 24 | 1024 | 16 | ~325M | [[TensorFlow]](https://drive.google.com/file/d/1a_OBYy6-4akWsk-9ciT5FAmsiwVONpvh/view?usp= sharing) | [[TensorFlow]](https://pan.baidu.com/s/1pxsS3almc90DPvMXH6BMYQ?pwd=s82t)<br/>(password: s82t) |
| **Chinese-LERT-base** | 12 | 768 | 12 | ~102M | [[TensorFlow]](https://drive.google.com/file/d/1SD0P5O9NCZTJ5qOzvJo7QGyAJapNM_YS/view?usp=sharing) | [ [TensorFlow]](https://pan.baidu.com/s/1_yb1jCDJ4s2P8OrF_5E_Tg?pwd=9jgi)<br/>(password: 9jgi) |
| **Chinese-LERT-small** | 12 | 256 | 4 | ~15M | [[TensorFlow]](https://drive.google.com/file/d/1CRyI58lhih5pDzajUbU6AFoFWFnJq9eA/view?usp=sharing) | [ [TensorFlow]](https://pan.baidu.com/s/1fBk3em8a5iCMwPLJEBq2pQ?pwd=4vuy)<br/>(password: 4vuy) |

> *Training corpus: Chinese Wikipedia, other encyclopedias, news, Q&A and other data, the total number of words is 5.4B, occupying about 20G disk space, the same as MacBERT.
>
> **Parameter number: only the transformer part is counted, and the parameter of the task heads is not included.

Take the TensorFlow version of `Chinese-LERT-base` as an example, after downloading, decompress the zip file to get:

````
chinese_lert_base_L-12_H-768_A-12.zip
|- lert_model.ckpt # model weights
|- lert_model.meta # model meta information
|- lert_model.index # model index information
|- lert_config.json # model parameters
|- vocab.txt # Vocabulary (same as Google original)
````

### PyTorch and TensorFlow 2 version

- TensorFlow (v2) and PyTorch version models can be downloaded through the 🤗 transformers model library.
- The PyTorch open-source version includes weights for the MLM parts, but does not include linguistic heads.

Download method: Click on any model to be downloaded → select the "Files and versions" tab → download the corresponding model file.

| Model | Model file size | Transformers model hub (supports MLM fill-in-the-blank interaction) |
| :------- | :---------: | :------------ |
| **Chinese-LERT-large** | ~1.2G | https://huggingface.co/hfl/chinese-lert-large |
| **Chinese-LERT-base** | ~400M | https://huggingface.co/hfl/chinese-lert-base |
| **Chinese-LERT-small** | ~60M | https://huggingface.co/hfl/chinese-lert-small |

## QuickLoad
Since the main body of LERT is still a BERT structure, users can easily call the LERT model using the [transformers library](https://github.com/huggingface/transformers).

**Note: All models in this directory are loaded using BertTokenizer and BertModel. **

````python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
````
The corresponding list of `MODEL_NAME` is as follows:

| Model name | MODEL_NAME |
| ---------------------- | ---------------------------- |
| Chinese-LERT-large | hfl/chinese-lert-large |
| Chinese-LERT-base | hfl/chinese-lert-base |
| Chinese-LERT-small | hfl/chinese-lert-small |

## Baselines
In our paper, the effect is tested on the following 10 tasks. Only some of them are shown in the GitHub directory, please refer to the paper for the full results.
- **Extractive Reading Comprehension** (2): [CMRC 2018 (Simplified Chinese)](https://github.com/ymcui/cmrc2018), [DRCD (Traditional Chinese)](https://github.com /DRCKnowledgeTeam/DRCD)
- **Text Classification** (6):
  - **Single Sentence** (2): [ChnSentiCorp](https://github.com/pengming617/bert_classification), [TNEWS](https://github.com/CLUEbenchmark/CLUE)
  - **Sentence pairs** (4): [XNLI](https://github.com/google-research/bert/blob/master/multilingual.md), [LCQMC](http://icrc.hitsz. edu.cn/info/1037/1146.htm), [BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html), [OCNLI](https://github.com /CLUEbenchmark/OCNLI)

- **Named Entity Recognition** (2): [MSRA-NER](), [People's Daily]()

In the experimental results table,

1. Outside the brackets is the maximum value of multiple finetunes, and inside the brackets is the average value.
2. Except for the *BERT* (Google's original BERT-base) model, the rest of the models are trained with the same amount of data.
3. RoBERTa-base and RoBERTa-large refer to [`RoBERTa-wwm-ext` and `RoBERTa-wwm-ext-large`](https://github.com/ymcui/Chinese-BERT-wwm) respectively.

### Reading Comprehension (CMRC 2018)

[**CMRC 2018**](https://github.com/ymcui/cmrc2018) is the Chinese machine reading comprehension data (extractive) released by the Harbin Institute of Technology Xunfei Joint Laboratory, in the same format as SQuAD. (Evaluation metrics: EM/F1)

| Model | Dev Set | Test Set | Challenge Set |
| :------- | :---------: | :---------: | :---------: |
| **↓ base model** ||||
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) |
| BERT-wwm-ext | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| RoBERTa-base | 67.4 (66.5) / 87.2 (86.5) | 72.6 (71.4) / 89.4 (88.8) | 26.2 (24.6) / 51.0 (49.1) |
| MacBERT-base|68.5 (67.3) / 87.9 (87.1)|73.2 (72.4) / 89.5 (89.2)|**30.2** (26.4) / 54.0 (52.2)|
| PERT-base |68.5 (68.1) / 87.2 (87.1)|72.8 (72.5) / 89.2 (89.0)|28.7 (**28.2**) / 55.4 (53.7)|
| **LERT-base** |**69.2 (68.4) / 88.1 (87.9)**|**73.5 (72.8) / 89.7 (89.4)**|27.7 (26.7) / **55.9 (54.6)** |
| **↓ large model** ||||
| RoBERTa-large | 68.5 (67.6) / 88.4 (87.9) | 74.2 (72.4) / 90.6 (90.0) | 31.5 (30.1) / 60.1 (57.5) |
| MacBERT-base|70.7 (68.6) / 88.9 (88.2)|74.8 (73.2) / 90.7 (90.1)|31.9 (29.6) / 60.2 (57.6)|
| PERT-base |**72.2 (71.0)** / 89.4 (88.8)|**76.8 (75.5)** / 90.7 (90.4)|**32.3 (30.9)** / 59.2 (58.1)|
| **LERT-base** |71.2 (70.5) / **89.5 (89.1)**|75.6 (75.1) / **90.9 (90.6)**|32.3 (29.7) / **61.2 (59.2)** |


### Single sentence text classification (ChnSentiCorp, TNEWS)

The following are the sentiment classification dataset ChnSentiCorp and news classification dataset TNEWS results. (Evaluation metric: Acc)
| Models | ChnSentiCorp - Dev Sets | TNEWS - Dev Sets |
| :------- | :---------: | :---------: |
| **↓ base model** |||
| BERT-wwm-ext | **95.4** (94.6) |57.0 (56.6)|
| RoBERTa-base | 94.9 (94.6) |57.4 (56.9)|
|MacBERT-base|95.2 (**94.8**)|57.4 (**57.1**)|
|PERT-base |94.0 (93.7)|56.7 (56.1)|
| **LERT-base** |94.9 (94.7)|**57.5 (57.1)**|
| **↓ large model** |||
| RoBERTa-large | **95.8** (94.9) |58.8 (58.4)|
|MacBERT-base|95.7 (**95.0**)|**59.0 (58.8)**|
|PERT-base |94.5 (94.0)|57.4 (57.2)|
| **LERT-base** |95.6 (94.9)|58.7 (58.5)|

### Sentence pair text classification (XNLI, OCNLI)

The following are the results of natural language inference on XNLI and OCNLI datasets. (Evaluation metric: Acc)

| Models | XNLI - Dev Set | OCNLI - Dev Set |
| :------- | :---------: | :---------: |
| **↓ base model** |||
| BERT-wwm-ext | 79.4 (78.6) |76.0 (75.3)|
| RoBERTa-base | 80.0 (79.2) |76.5 (76.0)|
|MacBERT-base|**80.3 (79.7)**|77.0 (76.5)|
|PERT-base |78.8 (78.1)|75.3 (74.8)|
| **LERT-base** |80.2 (79.5)|**78.2 (77.5)**|
| **↓ large model** |||
| RoBERTa-large | 82.1 (81.3) |78.5 (78.2)|
|MacBERT-base|**82.4 (81.8)**|79.0 (78.7)|
|PERT-base |81.0 (80.4)|78.1 (77.8)|
| **LERT-base** |81.7 (81.2)|**79.4 (78.9)**|

### Named Entity Recognition (MSRA, PD)

The following are the MSRA (test set) and People's Daily dataset (dev set) results. (Evaluation metric: F score)

| Models | MSRA-Test Set|PD-Development Set|
| :------- | :---------: | :---------: |
| **↓ base model** |||
| BERT-wwm-ext | 95.3 (94.9) |95.3 (95.1)|
| RoBERTa-base | 95.5 (95.1) |95.1 (94.9)|
|MacBERT-base|95.3 (95.1)|95.2 (94.9)|
|PERT-base |95.6 (95.3)|95.3 (95.1)|
| **LERT-base** |**95.7 (95.4)**|**95.6 (95.4)**|
| **↓ large model** |||
| RoBERTa-large | 95.5 (95.5) |95.7 (95.4)|
|MacBERT-base|96.2 (95.9)|95.8 (95.7)|
|PERT-base |96.2 (96.0)|96.1 (95.8)|
| **LERT-base** |**96.3 (96.0)**|**96.3 (96.0)**|


### Results for Small model
TBA

## FAQ
**Q1: Why doesn't the PyTorch version include linguistic heads?**
A1: The PyTorch version model is converted from the original TF weights. In order to directly use BERT-related interface to read the LERT model, the PyTorch version only includes the weights of the Transformer+MLM part. For the full version of the model, please download the TF 1.x version of the model. In addition, it should be noted that if you need to use it directly in downstream tasks or perform further pre-training, you do not need the weight of linguistic heads.

**Q2: Is there an English model for download? **
A2: There is no plan to train in English at the moment.


## FAQ
TBA


## Follow us
Welcome to follow the official WeChat account of HFL to keep up with the latest technical developments.

![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)


## Issues
If you have questions, please submit them in a GitHub Issue.

- Before submitting an issue, please check whether the FAQ can solve the problem, and it is recommended to check whether the previous issue can solve your problem.
- Duplicate and unrelated issues will be handled by [stable-bot](stale · GitHub Marketplace).
- We will try our best to answer your questions, but there is no guarantee that your questions will be answered.
- Politely ask questions and build a harmonious discussion community
