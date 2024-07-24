# ECCV2024:Any Target Can be Offense: Adversarial Example Generation via Generalized Latent Infection

This repository is the official implementation of our paper [Any Target Can be Offense: Adversarial Example Generation via Generalized Latent Infection](https://arxiv.org/abs/2407.12292). In this paper, we propose **Generalized Adversarial attacKER (GAKer)**, which is able to construct adversarial examples to any target class. The core idea behind GAKer is to craft a latently infected representation during adversarial example generation.

## Requirements


1. create environment

```bash
conda create -n gaker python==3.8
conda activate gaker

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge 

pip install ....

```


2. Datasets: 
  Download ImageNet Dataset


## Generalized Adversarial attacKER (GAKer)

1. step1:

```bash
python a.py
```

2. step2

```bash
python 2.py 
```
The results are stored in `./adv/`.


## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{sun2024targetoffenseadversarialexample,
      title={Any Target Can be Offense: Adversarial Example Generation via Generalized Latent Infection}, 
      author={Youheng Sun and Shengming Yuan and Xuanhan Wang and Lianli Gao and Jingkuan Song},
      year={2024},
Booktitle = {ECCV},

}
```
