# On Sparse Modern Hopfield Model
This is the code of the paper [On Sparse Modern Hopfield Model](https://arxiv.org/pdf/2309.12673.pdf). You can use this repo to reproduce the results in the paper.

## Environmental Setup

You can set up the experimental environment by running the following command line:

```shell
$ conda create -n sparse_hopfield python=3.8
$ conda activate sparse_hopfield
$ pip3 install -r requirements.txt
```

## Experimental Validation of Theoretical Results

### Dataset Preparation

```bash

```

### Plotting

```shell
$ python3 Plotting.py
```



## Multiple Instance Learning(MIL) Tasks

### Synthetic Experiments

```bash
$ python3 bit_numerical.py --model <MODEL> --exp <EXP>
```

Argument options 
* `model`: pooling, hopfield
* `exp`: bag_size, sparsity, both

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
@misc{hu2023sparse,
      title={On Sparse Modern Hopfield Model}, 
      author={Jerry Yao-Chieh Hu and Donglin Yang and Dennis Wu and Chenwei Xu and Bo-Yu Chen and Han Liu},
      year={2023},
      eprint={2309.12673},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
