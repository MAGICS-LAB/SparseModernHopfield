# On Sparse Modern Hopfield Model
This is the code of the paper [On Sparse Modern Hopfield Model](https://arxiv.org/pdf/2309.12673.pdf). You can use this repo to reproduce the results of our method.

## Environmental Setup

You can set up the experimental environment by running the following command line:

```shell
$ conda create -n sparse_hopfield python=3.8
$ conda activate sparse_hopfield
$ pip3 install -r requirements.txt
```

## Experimental Validation of Theoretical Results

### Plotting

```shell
$ python3 Plotting.py
```

## Multiple Instance Learning(MIL) Tasks

### MNIST MIL Experiments

(There might be some potential instability in bit pattern exps so please refer to the MNIST MIL exp for now.)

```shell
$ python3 mnist_mil_main.py --bag_size <BAG_SIZE>
```

Bag Size 5 (default setting)
<p float="left">
<p align="middle">
  <img src="/imgs/train_acc_5.png" width="24%" />
  <img src="/imgs/test_acc_5.png" width="24%" /> 
  <img src="/imgs/train_loss_5.png" width="24%" />
  <img src="/imgs/test_loss_5.png" width="24%" />
</p>
</p>

Bag Size 20 (default setting)
<p float="left">
<p align="middle">
  <img src="/imgs/train_acc_20.png" width="24%" />
  <img src="/imgs/test_acc_20.png" width="24%" /> 
  <img src="/imgs/train_loss_20.png" width="24%" />
  <img src="/imgs/test_loss_20.png" width="24%" />
</p>
</p>

Bag Size 30 (default setting)
<p float="left">
<p align="middle">
  <img src="/imgs/train_acc_30.png" width="24%" />
  <img src="/imgs/test_acc_30.png" width="24%" /> 
  <img src="/imgs/train_loss_30.png" width="24%" />
  <img src="/imgs/test_loss_30.png" width="24%" />
</p>
</p>

Bag Size 50 (default setting)
<p float="left">
<p align="middle">
  <img src="/imgs/train_acc_50.png" width="24%" />
  <img src="/imgs/test_acc_50.png" width="24%" /> 
  <img src="/imgs/train_loss_50.png" width="24%" />
  <img src="/imgs/test_loss_50.png" width="24%" />
</p>
</p>

Bag Size 80 (default setting)
<p float="left">
<p align="middle">
  <img src="/imgs/train_acc_80.png" width="24%" />
  <img src="/imgs/test_acc_80.png" width="24%" /> 
  <img src="/imgs/train_loss_80.png" width="24%" />
  <img src="/imgs/test_loss_80.png" width="24%" />
</p>
</p>

Bag Size 100 (dropout = 0.1)
<p float="left">
<p align="middle">
  <img src="/imgs/train_acc_100.png" width="24%" />
  <img src="/imgs/test_acc_100.png" width="24%" /> 
  <img src="/imgs/train_loss_100.png" width="24%" />
  <img src="/imgs/test_loss_100.png" width="24%" />
</p>
</p>


### Real-World MIL Tasks

#### Dataset preparation

Download and upzip the dataset

```bash
$ wget http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz 
$ wget http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz 
$ tar zxvf ./MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz 
```
#### Training and Evaluation

```bash
$ python3 real_world_mil.py --dataset <DATASET> --mode <MODE>
```

Argument options 
* `dataset`: fox, tiger, ucsb, elephant
* `mode`: sparse, standard
* `cpus_per_trial`: how many cpus do u want to use for a single run (set this up carefully for hyperparameter tuning)
* `gpus_per_trial`: how many gpus do u want to use for a single run (set this up carefully for hyperparameter tuning) (no larger than 1)
* `gpus_id`: specify which gpus u want to use (e.g. `--gpus_id=0, 1` means cuda:0 and cuda:1 are used for this script)

## Citations
Please consider citing our paper in your publications if it helps. Here is the bibtex:

```
@inproceedings{
hu2023on,
title={On Sparse Modern Hopfield Model},
author={Jerry Yao-Chieh Hu and Donglin Yang and Dennis Wu and Chenwei Xu and Bo-Yu Chen and Han Liu},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=eCgWNU2Imw}
}
```
