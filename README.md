# AL-ARCF

Pytorch implementation for codes in "An Active Learning-based Alternative Reinforcement Contextual Information Fusion Model for Multimodal Sentiment Analysis (AL-ARCF)"(https://ieeexplore.ieee.org/document/11122316)

![Overall_Framework](D:\博士课题\Xiaojiang_Paper\7. HCF\imgs\Overall_Framework.png)

# Prepare

## Datasets

Download the pkl file (https://drive.google.com/drive/folders/1_u1Vt0_4g0RLoQbdslBwAdMslEdW1avI?usp=sharing). Put it under the "./dataset" directory.

## Pre-trained language model

Download the SentiLARE language model files (https://drive.google.com/file/d/1onz0ds0CchBRFcSc_AkTLH_AZX_iNTjO/view?usp=share_link), and then put them into the "./pretrained-model/sentilare_model" directory.

# Run

''' python train.py '''

# Paper

Please cite our paper if you find our work useful for your research:

```
@ARTICLE{11122316,
  author={He, Xiaojiang and Pan, Yushan and Chen, Yangbin and Li, Zuhe and Xu, Zhijie and Yang, Chenguang and Wang, Kaiwei},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={An Active Learning-Based Alternative Reinforcement Contextual Information Fusion Model for Multimodal Sentiment Analysis}, 
  year={2025},
  volume={33},
  number={},
  pages={3537-3552},
  keywords={Feature extraction;Sentiment analysis;Data mining;Active learning;Costs;Context modeling;Data models;Visualization;Speech processing;Redundancy;Active learning;curriculum learning;global contextual feature;multimodal sentiment analysis},
  doi={10.1109/TASLPRO.2025.3597474}}

```

