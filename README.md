# **WTHMDA**

## Contents

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Installation](#installation)
- [Data preprocess](#model-training)
- [Model training and prediction ](#classification )
- [Citation](#citation)
- [Contact](#contact)

## Introduction

In this study, we propose a novel deep learning framework named WTHMDA (Weighted Taxonomic Heterogeneous-network based Microbe-Disease Association), which leverages a weighted graph convolution network and microbial taxonomy common tree for microbe-disease association prediction.

## Package requirement

- torch==1.9.0
- dgl==0.9.1
- ete3==3.1.3
- gensim==4.0.1

## Installation

```
sh init.sh
```

Then all tools are located at ‘bin’ folder:
get_disease_feature.py // To generate disease feature utilizing weighted Deepwalk
get_microbe_feature.py // To generate microbe feature utilizing weightedDeepwalk
get_edges.py // To generate directed microbe links and four types of weighted microbe ontological similarities
WTHMDA_run.py // For model training and prediction

## Data preprocess

To generate disease feature utilizing weighted Deepwalk, one file are required as input including integrated disease functional similarities.  
a. integrated disease functional similarities (required)

| 1    | 0.21 | 0.52 | 0.11 |
| ---- | ---- | ---- | ---- |
| 0.21 | 1    | 0    | 0.44 |
| 0.52 | 0    | 1    | 0.87 |
| 0.11 | 0.44 | 0.87 | 1    |

To generate microbe feature utilizing weighted Deepwalk, one file are required as input including integrated microbe functional similarities.
a. integrated microbe functional similarities (required)

| 1    | 0.75 | 0.9  | 0.43 |
| ---- | ---- | ---- | ---- |
| 0.75 | 1    | 0.71 | 0.1  |
| 0.9  | 0.71 | 1    | 0.53 |
| 0.43 | 0.1  | 0.53 | 1    |

To generate microbe feature utilizing weighted Deepwalk, one file are required as input including integrated microbe functional similarities.
a. integrated microbe functional similarities (required)

| 0   | Abiotrophia |
| --- | ----------- |
| 1   | Bacteroides |
| 2   | Caloramator |
| 3   | Dermacoccus |

You can specify IDFS, IMFS and microbial lists by "--dis", "--mis" and "--mlist" respectively. We set up an example dataset(example) in the "data" folder for a quick start:

```
cd data

python ../bin/get_disease_feature.py --dis example/disease_I_S.csv
python ../bin/get_microbe_feature.py --mis example/microbe_I_S.csv
python ../bin/get_edges.py --mlist example/microbe_list.xlsx
```

## Model training and prediction

To train WTHMDA model and prediction, two files are required as input including train_edge and test_edge. 
a. train_edge (required)

| disease | microbe | label |
| ------- | ------- | ----- |
| 1       | 108     | 1     |
| 4       | 43      | 0     |
| 36      | 230     | 1     |

b. test_edge (required)

| disease | microbe | label |
| ------- | ------- | ----- |
| 3       | 165     | 1     |
| 15      | 62      | 1     |
| 14      | 42      | 0     |

You can assign train data and test data  by '--train'  and '--test', respectively. In addition, you can specify the output path of model '--o'. We set an example dataset in ‘data’ folder for quick start. To train a Meta-Spec model:
    cd data
    python ../bin/meta_spec_train.py --microbe train_microbe_data.csv --host train_hosts_data.csv --label train_labels.csv --o out

Then, you can predict the status of microbiomes using the model generated by the training procedure. We set an example dataset(example) in ‘data/’ folder for quick start.

```
cd data

python ../bin/train_test.py --train example/train_edge.csv --test example/test_edge.csv --o ../result/ 
```

For convenience, you can run the processes above by running the example.sh in folder 'data/'.

```
    cd data
    chmod a+x example.sh
    ./example.sh
```

Other settings can be seen in the config.py.

## Citation

## Contact

All problems please contact WTHMDA development team: 
**Xiaoquan Su**&nbsp;&nbsp;&nbsp;&nbsp;Email: suxq@qdu.edu.cn
