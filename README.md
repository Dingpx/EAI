# Expressive Forecasting of 3D Whole-body Human Motions (arxiv'22)

## 🌍 [Project Page]() | 📝 [ArXiv]() | 🎥 [Video]() | 💿 [Dataset]()

<br/>

> Forecasting Characteristic 3D Poses of Human Actions <br />
> [Pengxiang Ding](https://dingpx.github.io/), [Qiongjie Cui](https://keras.me/), [Hongwei Yi](https://xyyhw.top/), [Mingyu Ding](https://dingmyu.github.io/),
[Debing Zhang](https://scholar.google.com/citations?user=4nL1cDEAAAAJ&hl=en), [Haofan Wang](https://haofanwang.github.io/), [Qiuhong Ke](https://scholar.google.com/citations?user=84qxdhsAAAAJ&hl=zh-CN), [Jun Liu](https://scholar.google.com/citations?user=Q5Ild8UAAAAJ&hl=zh-CN)<br/>



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
- The whole process of [GRAB](https://grab.is.tue.mpg.de/)  will be updated
- or you can download our [processed data]()


## Training
Run `$ bash run_train.sh`.

##  Evaluation
Run `$ bash run.sh`.


## License
EAI is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
See LICENSE.txt for more details.
