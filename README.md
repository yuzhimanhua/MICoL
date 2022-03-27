# Metadata-Induced Contrastive Learning for Zero-Shot Multi-Label Text Classification
**We are still finalizing this code repository. The current version should be enough to reproduce results in our paper. We will release an easier-to-run version before the WWW 2022 conference.**

## Installation
For training, GPUs are strongly recommended. We use one NVIDIA V100 to run each experiment.

### Dependency
We use PyTorch and HuggingFace transformers to build the model. The dependencies are summarized in the file ```requirements.txt```. You can install them like this:
```
pip3 install -r requirements.txt
```

## Setup
To reproduce the results in our paper, you need to first download the [**datasets**](https://drive.google.com/file/d/1FD0ddpMmWMFDdk1SwbEZ3xy93b1NvbBz/view?usp=sharing). Two datasets are used in our paper: **MAG-CS** and **PubMed**. Once you unzip the downloaded file (i.e., ```MICoL.zip```), you can see **four** folders: ```MAG/``` is the dataset folder of MAG-CS; ```PubMed/``` is the dataset folder of PubMed; ```scibert_scivocab_uncased/``` is the pre-trained SciBERT model; ```BM25/``` contains the candidate labels selected from the retrieval stage.

Put these four folders under the main directory ```./```. Then the following running script can be used to prepare training and testing data, where training document pairs are generated from metadata for contrastive learning.
```
python prepare_test.py
./gen_train.sh
```

## Training and Prediction
Please use the following script to conduct training and prediction.
```
./run.sh
```
You can change the dataset (mag or pubmed), the meta-path/meta-graph (10 choices), and the architecture (bi or cross) in the script.

## Evaluation
Please use the following script to perform evaluation.
```
python patk.py
```
It will output P@_k_ and NDCG@_k_ scores (_k_=1,3,5). 
