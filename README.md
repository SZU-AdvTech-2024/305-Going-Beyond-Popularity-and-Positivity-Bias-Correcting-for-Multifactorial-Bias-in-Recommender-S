# MultifactorialBias
This repository contains the code used for the experiments in ["Going Beyond Popularity and Positivity Bias: Correcting for Multifactorial Bias in Recommender Systems"](https://doi.org/10.1145/3626772.3657749).

## Citation
If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our SIGIR 2024 paper:
```
@inproceedings{huang-2024-going,
author = {Huang, Jin and Oosterhuis, Harrie and Mansoury, Masoud and van Hoof, Herke and de Rijke, Maarten},
booktitle = {SIGIR 2024: The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
month = {July},
publisher = {ACM},
title = {Going Beyond Popularity and Positivity Bias: Correcting for Multifactorial Bias in Recommender Systems},
year = {2024}}
```

## Required packages
You can install conda and then create Python 3.9 Conda environment. 
Create the environment from the ```environment.yml``` and activate it:
```
$ conda env create -f environment.yml
$ conda activate Multifactorial
```

## Reproducing Experiments on Real-world Data
Our experimental analysis is conducted on real-world datasets: the Yahoo!R3 and Coat datasets. The preprocessed data can be found [here](https://drive.google.com/file/d/1jg9BE7ZoR4ehXifNeaEJx5AVNYRANnr6/view?usp=sharing). Please download it, unzip it, and then move the obtained folders ```./data``` and ```./propensities_gen_by_mf``` to the main directory of the project.

   

### Concurrent optimization
Reproducing the results of methods - MF, MF-IPS $^{Pop}$, MF-IPS $^{Pos}$, MF-IPS $^{MF}$, and MF-IPS $^{Mul}$ optimized by the concurrent gradient descent method, \
on the Yahoo!R3 dataset:
```
nohup bash -c "time python mf-concurrent.py --dataset_name yahoo --debiasing none --lr 1e-4 --reg 1e-4 --dim 128"> output/concurrent/yahoo_none_out 2>&1 &
nohup bash -c "time python mf-concurrent.py --dataset_name yahoo --debiasing popularity --lr 1e-5 --reg 1e-4 --dim 64"> output/concurrent/yahoo_popularity_out 2>&1 &
nohup bash -c "time python mf-concurrent.py --dataset_name yahoo --debiasing positivity --lr 1e-4 --reg 1e-4 --dim 16"> output/concurrent/yahoo_positivity_out 2>&1 &
nohup bash -c "time python mf-concurrent.py --dataset_name yahoo --debiasing mf --lr 1e-5 --reg 1e-4 --dim 64"> output/concurrent/yahoo_mf_out 2>&1 &
nohup bash -c "time python mf-concurrent.py --dataset_name yahoo --debiasing multifactorial --lr 1e-5 --reg 1e-4 --dim 32"> output/concurrent/yahoo_multifactorial_out 2>&1 &


$ python mf-concurrent.py --dataset_name yahoo --debiasing none --lr 1e-4 --reg 1e-4 --dim 128
$ python mf-concurrent.py --dataset_name yahoo --debiasing popularity --lr 1e-5 --reg 1e-4 --dim 64
$ python mf-concurrent.py --dataset_name yahoo --debiasing positivity --lr 1e-4 --reg 1e-4 --dim 16
$ python mf-concurrent.py --dataset_name yahoo --debiasing mf --lr 1e-5 --reg 1e-4 --dim 64
$ python mf-concurrent.py --dataset_name yahoo --debiasing multifactorial --lr 1e-5 --reg 1e-4 --dim 32
```
on the Coat dataset:
```
nohup bash -c "time python3 mf-concurrent.py --dataset_name coat --debiasing none --lr 1e-4 --reg 1e-7 --dim 16"> output/concurrent/coat_none_out 2>&1 &
nohup bash -c "time python3 mf-concurrent.py --dataset_name coat --debiasing popularity --lr 1e-4 --reg 1e-3 --dim 64"> output/concurrent/coat_popularity_out 2>&1 &
nohup bash -c "time python3 mf-concurrent.py --dataset_name coat --debiasing positivity --lr 1e-5 --reg 1e-5 --dim 128"> output/concurrent/coat_positivity_out 2>&1 &
nohup bash -c "time python3 mf-concurrent.py --dataset_name coat --debiasing mf --lr 1e-4 --reg 1e-3 --dim 128"> output/concurrent/coat_mf_out 2>&1 &
nohup bash -c "time python3 mf-concurrent.py --dataset_name coat --debiasing multifactorial --lr 1e-4 --reg 1e-3 --dim 128"> output/concurrent/coat_multifactorial_out 2>&1 &

$ python mf-concurrent.py --dataset_name coat --debiasing none --lr 1e-4 --reg 1e-7 --dim 16
$ python mf-concurrent.py --dataset_name coat --debiasing popularity --lr 1e-4 --reg 1e-3 --dim 64
$ python mf-concurrent.py --dataset_name coat --debiasing positivity --lr 1e-5 --reg 1e-5 --dim 128
$ python mf-concurrent.py --dataset_name coat --debiasing mf --lr 1e-4 --reg 1e-3 --dim 128
$ python mf-concurrent.py --dataset_name coat --debiasing multifactorial --lr 1e-4 --reg 1e-3 --dim 128
```

### Alternating optimization


Reproducing the results of methods - MF, MF-IPS $^{Pop}$, MF-IPS $^{Pos}$, MF-IPS $^{MF}$, and MF-IPS $^{Mul}$ optimized by the alternating gradient descent method, \
on the Yahoo!R3 dataset:
```
nohup bash -c "time python3 mf-alternating.py --dataset_name yahoo --debiasing none --lr 1e-5 --reg 1e-4 --dim 128"> output/alternating/yahoo_none_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name yahoo --debiasing popularity --lr 1e-5 --reg 1e-4 --dim 32"> output/alternating/yahoo_popularity_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name yahoo --debiasing positivity --lr 1e-5 --reg 1e-4 --dim 128"> output/alternating/yahoo_positivity_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name yahoo --debiasing mf --lr 1e-5 --reg 1e-4 --dim 32"> output/alternating/yahoo_mf_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name yahoo --debiasing multifactorial --lr 1e-5 --reg 1e-4 --dim 32"> output/alternating/yahoo_multifactorial_out 2>&1 &

$ python mf-alternating.py --dataset_name yahoo --debiasing none --lr 1e-5 --reg 1e-4 --dim 128
$ python mf-alternating.py --dataset_name yahoo --debiasing popularity --lr 1e-5 --reg 1e-4 --dim 32
$ python mf-alternating.py --dataset_name yahoo --debiasing positivity --lr 1e-5 --reg 1e-4 --dim 128
$ python mf-alternating.py --dataset_name yahoo --debiasing mf --lr 1e-5 --reg 1e-4 --dim 32
$ python mf-alternating.py --dataset_name yahoo --debiasing multifactorial --lr 1e-5 --reg 1e-4 --dim 32
```
on the Coat dataset:
```
nohup bash -c "time python3 mf-alternating.py --dataset_name coat --debiasing none --lr 1e-4 --reg 1e-3 --dim 128"> output/alternating/coat_none_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name coat --debiasing popularity --lr 1e-5 --reg 1e-3 --dim 128"> output/alternating/coat_popularity_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name coat --debiasing positivity --lr 1e-5 --reg 1e-6 --dim 128"> output/alternating/coat_positivity_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name coat --debiasing mf --lr 1e-4 --reg 1e-3 --dim 128"> output/alternating/coat_mf_out 2>&1 &
nohup bash -c "time python3 mf-alternating.py --dataset_name coat --debiasing multifactorial --lr 1e-3 --reg 1e-3 --dim 128"> output/alternating/coat_multifactorial_out 2>&1 &

$ python mf-alternating.py --dataset_name coat --debiasing none --lr 1e-4 --reg 1e-3 --dim 128
$ python mf-alternating.py --dataset_name coat --debiasing popularity --lr 1e-5 --reg 1e-3 --dim 128
$ python mf-alternating.py --dataset_name coat --debiasing positivity --lr 1e-5 --reg 1e-6 --dim 128
$ python mf-alternating.py --dataset_name coat --debiasing mf --lr 1e-4 --reg 1e-3 --dim 128
$ python mf-alternating.py --dataset_name coat --debiasing multifactorial --lr 1e-3 --reg 1e-3 --dim 128
```


<!-- ### RQ2: How do varying smoothing parameters and our alternating gradient descent approach affect our multifactorial method? -->

The results of VAE models on the Yahoo!R3 and Coat datasets can be reproduced by using:
```
nohup bash -c "time python3 mf-concurrent.py --dataset_name yahoo --CF_model VAE --debiasing none --lr 1e-5 --reg 1e-7"> output/yahoo_VAE_out 2>&1 &
nohup bash -c "time python3 mf-concurrent.py --dataset_name coat --CF_model VAE --debiasing none --lr 1e-5 --reg 1e-3"> output/coat_VAE_out 2>&1 &


$ python mf-concurrent.py --dataset_name yahoo --CF_model VAE --debiasing none --lr 1e-5 --reg 1e-7
$ python mf-concurrent.py --dataset_name coat --CF_model VAE --debiasing none --lr 1e-5 --reg 1e-3
```

## Reproducing Experiments on Synthetic Data
We further perform an extensive simulation-based experimental analysis where the effect of each of the two factors is varied and answer the research question: Can our multifactorial method MF-IPS $^{Mul}$ robustly mitigate the effect of selection bias in scenarios where the effect of two factors on bias is varied?

Our simulated multifactorial propensity is then simply a linear interpolation between $\rho^{(\text{R})}$ which is only dependent on the rating values, and $\rho^{(\text{I})}$ which is only dependent on the items: $$P(o=1 \mid y=r, i) = \gamma \rho^{(\text{R})}_r + (1 - \gamma) \rho^{(\text{I})}_i,$$
where $\gamma \in [0, 1]$ controls the effect of each factor on the selection bias.

Reproducing the results of methods - MF, MF-IPS $^{GT}$, MF-IPS $^{Pop}$, MF-IPS $^{Pos}$, and MF-IPS $^{Mul}$ optimized by the alternating gradient descent method when $\gamma = 0.5$:

```

nohup bash -c "time python3 semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=none --lr=0.0001 --reg=0.0001 --dim=32 --ALS=True"> output/semi-synthetic_none_out 2>&1 &
nohup bash -c "time python3 semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=GT --lr=0.0001 --reg=1e-07 --dim=128 --ALS=True"> output/semi-synthetic_GT_out 2>&1 &
nohup bash -c "time python3 semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=positivity --lr=1e-05 --reg=0.0001 --dim=32 --ALS=True"> output/semi-synthetic_positivity_out 2>&1 &
nohup bash -c "time python3 semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=popularity --lr=0.0001 --reg=0.0001 --dim=32 --ALS=True"> output/semi-synthetic_popularity_out 2>&1 &
nohup bash -c "time python3 semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=multifactorial --lr=0.0001 --reg=0.0001 --dim=16 --ALS=True"> output/semi-synthetic_multifactorial_out 2>&1 &



$ python semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=none --lr=0.0001 --reg=0.0001 --dim=32 --ALS=True
$ python semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=GT --lr=0.0001 --reg=1e-07 --dim=128 --ALS=True
$ python semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=positivity --lr=1e-05 --reg=0.0001 --dim=32 --ALS=True
$ python semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=popularity --lr=0.0001 --reg=0.0001 --dim=32 --ALS=True
$ python semi-synthetic_data_bias.py --mul_alpha=0.5 --debiasing=multifactorial --lr=0.0001 --reg=0.0001 --dim=16 --ALS=True
```
The hyperparameter choices for scenario when $\gamma \in [0.0, 0.1, \ldots, 1.0]$ can be found in file ```parameters-semi-data.txt```.