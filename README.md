# On Sparse Modern Hopfield Model
This is the code of the paper [On Sparse Modern Hopfield Model](https://arxiv.org/pdf/2309.12673.pdf). You can use this repo to reproduce the results in the paper.

## Citations
Please consider citing our paper in your publications if it helps. Here is the bibtex:

```
@inproceedings{
  hu2023sparse,
  title={On Sparse Modern Hopfield Model},
  author={Jerry Yao-Chieh Hu and Donglin Yang and Dennis Wu and Chenwei Xu and Bo-Yu Chen and Han Liu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://arxiv.org/abs/2309.12673}
}
```

## Environmental Setup

You can set up the experimental environment by running the following command line:

```shell
$ conda create -n sparse_hopfield python=3.8
$ conda activate sparse_hopfield
$ pip3 install -r requirements.txt
```

## Examples

In ```layers.py```, we have implemented the general sparse Hopfield, dense Hopfield and sparse Hopfield.
To use it, see below

```python
dense_hp = HopfieldPooling(
    d_model=d_model,
    n_heads=n_heads,
    mix=True,
    update_steps=update_steps,
    dropout=dropout,
    mode="softmax",
    scale=scale,
    num_pattern=num_pattern) # Dense Hopfield

sparse_hp = HopfieldPooling(
    d_model=d_model,
    n_heads=n_heads,
    mix=True,
    update_steps=update_steps,
    dropout=dropout,
    mode="sparsemax",
    scale=scale,
    num_pattern=num_pattern) # Sparse Hopfield

entmax_hp = HopfieldPooling(
    d_model=d_model,
    n_heads=n_heads,
    mix=True,
    update_steps=update_steps,
    dropout=dropout,
    mode="entmax",
    scale=scale,
    num_pattern=num_pattern) # Hopfield with Entmax-15

gsh_hp = HopfieldPooling(
    d_model=d_model,
    n_heads=n_heads,
    mix=True,
    update_steps=update_steps,
    dropout=dropout,
    mode="gsh",
    scale=scale,
    num_pattern=num_pattern) # Generalized Sparse Hopfield with learnable alpha
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


## Acknowledgment

The authors would like to thank the anonymous reviewers and program chairs for constructive comments.

JH is partially supported by the Walter P. Murphy Fellowship.
HL is partially supported by NIH R01LM1372201, NSF CAREER1841569, DOE DE-AC02-07CH11359, DOE LAB 20-2261 and a NSF TRIPODS1740735.
This research was supported in part through the computational resources and staff contributions provided for the Quest high performance computing facility at Northwestern University which is jointly supported by the Office of the Provost, the Office for Research, and Northwestern University Information Technology.
The content is solely the responsibility of the authors and does not necessarily represent the official
views of the funding agencies.

The experiments in this work benefit from the following open-source codes:
* Ramsauer, Hubert, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber et al. "Hopfield networks is all you need." arXiv preprint arXiv:2008.02217 (2020). https://github.com/ml-jku/hopfield-layers
* Martins, Andre, and Ramon Astudillo. "From softmax to sparsemax: A sparse model of attention and multi-label classification." In International conference on machine learning, pp. 1614-1623. PMLR, 2016. https://github.com/KrisKorrel/sparsemax-pytorch
* Correia, Gonçalo M., Vlad Niculae, and André FT Martins. "Adaptively sparse transformers." arXiv preprint arXiv:1909.00015 (2019). https://github.com/deep-spin/entmax & https://github.com/prajjwal1/adaptive_transformer
* Ilse, Maximilian, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." In International conference on machine learning, pp. 2127-2136. PMLR, 2018. https://github.com/AMLab-Amsterdam/AttentionDeepMIL
* Zhang, Yunhao, and Junchi Yan. "Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting." In The Eleventh International Conference on Learning Representations. 2022. https://github.com/Thinklab-SJTU/Crossformer
* Millidge, Beren, Tommaso Salvatori, Yuhang Song, Thomas Lukasiewicz, and Rafal Bogacz. "Universal hopfield networks: A general framework for single-shot associative memory models." In International Conference on Machine Learning, pp. 15561-15583. PMLR, 2022. https://github.com/BerenMillidge/Theory_Associative_Memory
