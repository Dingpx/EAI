# Expressive Forecasting of 3D Whole-body Human Motions (AAAI2024)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.11972)

Pengxiang Ding, Qiongjie Cui, Min Zhang, Mengyuan Liu, Haofan Wang, Donglin Wang




## Abstract
Human motion forecasting, with the goal of estimating future human behavior over a period of time, is a fundamental task in many real-world applications.
Existing work typically concentrates on foretelling the major joints of the human body without considering the delicate movements of the human hands.
In practical applications, hand gestures play an important role in human communication with the real world, and express the primary intentions of human beings.
In this work, we propose a new Encoding-Alignment-Interaction (EAI) framework to address expressive forecasting of 3D whole-body human motions, which aims to predict coarse- (body joints) and fine-grained (gestures) activities cooperatively.
To our knowledge, this meaningful topic has not been explored before.
Specifically, our model mainly involves two key constituents: cross-context alignment (XCA) and cross-context interaction (XCI).
Considering the heterogeneous information within the whole-body, the former aims to align the latent features of various human components, while the latter focuses on effectively capturing the 
context interaction among the human components. 
We conduct extensive experiments on a newly-introduced large-scale benchmark and achieve state-of-the-art performance.


## Installation
1. Clone this repository   
`$ git clone https://github.com/Dingpx/EAI.git`

2. Initialize conda environment    
`$ conda env create -f requirement.yaml`

## Datasets
### GRAB data
Updated: You can download our [processed data](https://drive.google.com/drive/folders/1o5wfHCkCTwOJrXs8dhGoRFoE1y4q42CO?usp=drive_link) 

TODO:
- The whole process of [GRAB](https://grab.is.tue.mpg.de/)  will be updated soon.



## Training
Run `$ bash run_train.sh`.

##  Evaluation
Run `$ bash run.sh`.


## Cite our work:
```
@article{ding2023expressive,
  title={Expressive Forecasting of 3D Whole-body Human Motions},
  author={Ding, Pengxiang and Cui, Qiongjie and Zhang, Min and Liu, Mengyuan and Wang, Haofan and Wang, Donglin},
  journal={arXiv preprint arXiv:2312.11972},
  year={2023}
}
```
