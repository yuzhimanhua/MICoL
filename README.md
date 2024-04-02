# Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the source code for [**Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification**](https://arxiv.org/abs/2202.05932).

## Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Running](#running)
- [Citation](#citation)


## Installation
For training, GPUs are strongly recommended. We use one NVIDIA V100 16GB GPU in our experiments.

### Dependency
The code is written in Python 3.6. We use PyTorch and HuggingFace transformers to build the model. The dependencies are summarized in the file ```requirements.txt```. You can install them like this:
```
pip3 install -r requirements.txt
```

## Quick Start
To reproduce the results in our paper, you need to first download the [**datasets**](https://drive.google.com/file/d/1FD0ddpMmWMFDdk1SwbEZ3xy93b1NvbBz/view?usp=sharing). Two datasets are used in our paper: **MAG-CS** and **PubMed**. Once you unzip the downloaded file (i.e., ```MICoL.zip```), you can see **three** folders: ```MAG/``` is the dataset folder of MAG-CS; ```PubMed/``` is the dataset folder of PubMed; ```scibert_scivocab_uncased/``` is the pre-trained SciBERT model. (The pre-trained SciBERT model is from [here](https://huggingface.co/allenai/scibert_scivocab_uncased/tree/main).)

Put the three folders under the main directory ```./```. Then you need to run the following scripts. 

### Input Preparation
Prepare training and testing data. Training document pairs are generated using document metadata for contrastive learning.
```
./prepare.sh
```

### Training, Testing, and Evaluation
```
./run.sh
```
P@_k_, NDCG@_k_, PSP@_k_, and PSN@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output. The prediction results can be found in ```./{dataset}_output/prediction_{architecture}.json``` (e.g., ```./MAG_output/prediction_cross.json```).

You can change the dataset (MAG or PubMed), the meta-path/meta-graph (10 choices, see the [Running](#running) section below), and the architecture (bi or cross) in the script.

## Data
Two datasets are used in our paper. They were first released in the [MATCH](https://github.com/yuzhimanhua/MATCH) project. In this paper, we combine the original training and validation sets together as our **unlabeled** training set (that being said, we do not know the labels of these documents, and we only utilize their text and metadata information); we use the original testing set as our testing set. Dataset statistics are listed below.
|  | MAG-CS | PubMed | 
|--|--------|--------|
| \# Training Documents (Unlabeled)   | 634,874   | 808,692   |
| \# Testing Documents                | 70,533    | 89,854    |
| \# Labels                           | 15,808\*  | 17,963    |
| \# Labels / Doc (in Testing)        | 5.59      | 7.80      |
| \# Words / Doc (in Testing)         | 126.55    | 199.14    |
| \# Authors (in Training)            | 762,259   | 2,068,411 |
| \# Paper-Author Edges (in Training) | 2,047,166 | 5,391,314 |
| \# Venues (in Training)             | 105       | 150       |
| \# Paper-Venue Edges (in Training)  | 634,874   | 808,692   |
| \# Paper->Paper Edges (in Training) | 1,219,234\*\* | 3,615,220\*\* |

\*: Originally, there were 15,809 labels in MAG-CS, but the label "Computer Science" is removed from all papers because it is trivial to predict.

\*\*: Both papers in the "Paper->Paper" edge need to appear in the training set.

### The MAG Dataset
After you download the [**datasets**](https://drive.google.com/file/d/1FD0ddpMmWMFDdk1SwbEZ3xy93b1NvbBz/view?usp=sharing), there are four input files in the ```MAG/``` folder: **```MAG_train.json```**, **```MAG_test.json```**, **```MAG_label.json```**, and **```MAG_candidates.json```**.

```MAG_train.json``` has text and metadata information of each training document. Each line is a json record representing one document. For example,
```
{
  "paper": "2150203549",
  "text": "probabilistic_logic group recommendation via information matching. increasingly web recommender_system face scenarios ...",
  "venue": "WWW",
  "author": [
    "2936723753", "2129454193", "2023330040", "2151628004"
  ],
  "reference": [
    "2051834357", "2164374228", "2172118809"
  ],
  "citation": [
    "2064702560", "2105547690"
  ],
  "label": [
    "119857082", "124101348", "197927960", "49937458", "58489278", "127705205", "109364899", "136764020", "557471498", "2776156558", "21569690"
  ]
}
```
Here, "paper" is the paper id; "reference" is the list of paper ids **the current paper cites**; "citation" is the list of paper ids **citing the current paper**. The "label" field in ```MAG_train.json``` is optional. We do not need label information of training documents in our training process.

```MAG_test.json``` has text and label information of each testing document. Each line is a json record representing one document. Its format is identical to a record in ```MAG_train.json```. However, now the "label" field is required for eveluation, and the metadata fields (i.e., "venue", "author", "reference", "citation") become optional. We do not need metadata information of testing documents during inference.

```MAG_label.json``` has the name and description of each label. Each line is a json record representing one label. For example,
```
{
  "label": "11045955",
  "name": [
    "elgamal encryption"
  ],
  "definition": "in cryptography, the elgamal encryption system is an asymmetric key encryption algorithm ...",
  "combined_text": "elgamal encryption. in cryptography, the elgamal encryption system is an asymmetric key encryption algorithm ..."
}
```
The "combined_text" field is the concatenation of "name" and "definition".

```MAG_candidates.json``` lists the candidate labels of each testing document (obtained by exact name matching and BM25 retrieval). Each line is a json record representing one testing document. For example,
```
{
  "paper": "2043341294",
  "label": [
    "60048249", "206134035", "2474386", "165297611", "195324797", "39890363", "70777604", "204321447"
  ],
  "predicted_label": [
    "70777604", "76482347", "192209626", "29808475", "206134035", "2474386", "166553842", "195324797", "165297611", "9628104", "39890363", "5147268", "60048249", "19768560"
  ]
}
```
Here, "label" is the list of ground-truth label ids of the document (identical to the "label" field in ```MAG_test.json```); "predicted_label" is the list of candidate label ids obtained by exact name matching and BM25 retrieval.

### The PubMed Dataset
There are four input files in the ```PubMed/``` folder: **```PubMed_train.json```**, **```PubMed_test.json```**, **```PubMed_label.json```**, and **```PubMed_candidates.json```**. Their format is the same as that of their counterpart in the MAG dataset.

## Running
The [Quick Start](#quick-start) section should be enough to reproduce the results in out paper. Here are some more details of running our code.

(1) In ```prepare.sh```, you can choose among different meta-paths/meta-graphs to generate positive training pairs. We support 10 choices of meta-paths/meta-graphs.
| Meta-path/Meta-graph | How you should write it in ```prepare.sh``` | 
|------------|----------------------|
| P->P       | ```metagraph=PR```   |
| P<-P       | ```metagraph=PC```   |
| P-A-P      | ```metagraph=PAP```  |
| P-V-P      | ```metagraph=PVP```  |
| P->P<-P    | ```metagraph=PRP```  |
| P<-P->P    | ```metagraph=PCP```  |
| P-(AA)-P   | ```metagraph=PAAP``` |
| P-(AV)-P   | ```metagraph=PAVP``` |
| P->(PP)<-P | ```metagraph=PRRP``` |
| P<-(PP)->P | ```metagraph=PCCP``` |

(2) In ```run.sh```, you can choose between two different architectures: Bi-Encoder (```architecture=bi```) and Cross-Encoder (```architecture=cross```).

## Citation
Our implementation is adapted from [Poly-Encoder](https://github.com/chijames/Poly-Encoder) and [CorNet](https://github.com/XunGuangxu/CorNet). If you find this repository useful, please cite the following paper:
```
@inproceedings{zhang2022metadata,
  title={Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification},
  author={Zhang, Yu and Shen, Zhihong and Wu, Chieh-Han and Xie, Boya and Hao, Junheng and Wang, Ye-Yi and Wang, Kuansan and Han, Jiawei},
  booktitle={WWW'22},
  pages={3162--3173},
  year={2022}
}
```
