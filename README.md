# Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification

This repository contains the source code for [**Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification**](https://arxiv.org/abs/2202.05932).

## Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Running](#running)
- [Citation](#citation)


## Installation
For training, GPUs are strongly recommended. We use one NVIDIA V100 16GB to run each experiment.

### Dependency
We use PyTorch and HuggingFace transformers to build the model. The dependencies are summarized in the file ```requirements.txt```. You can install them like this:
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
TBD

## Running
TBD

## Citation
Our implementation is adapted from [Poly-Encoder](https://github.com/chijames/Poly-Encoder). If you find the implementation useful, please cite the following paper:
```
@article{zhang2022metadata,
  title={Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification},
  author={Zhang, Yu and Shen, Zhihong and Wu, Chieh-Han and Xie, Boya and Hao, Junheng and Wang, Ye-Yi and Wang, Kuansan and Han, Jiawei},
  journal={arXiv preprint arXiv:2202.05932},
  year={2022}
}
```
