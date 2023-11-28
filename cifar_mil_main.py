import argparse
import json
from cifar_mil_trainer import *
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def get_args():

    parser = argparse.ArgumentParser(description='MNIST MIL benchmarks:')

    parser.add_argument("--project_name", default="MNIST-MIL")
    parser.add_argument('--wandb', default=False, type=bool)

    # Model params
    parser.add_argument('--mode', default="softmax", choices=["softmax", "entmax", "sparsemax", "gsh"])
    parser.add_argument('--d_model', default=1024, type=int)
    parser.add_argument('--input_size', default=3*32*32, type=int)
    parser.add_argument('--model', default="pooling", type=str)
    parser.add_argument('--num_pattern', default=20, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--scale', default=0.01)
    parser.add_argument('--update_steps', default=1, type=int)
    parser.add_argument('--dropout', default=0.7, type=float)

    # Training params
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--seed', default=1111, type=int)

    # Data params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--train_size', default=10000, type=int)
    parser.add_argument('--test_size', default=5000, type=int)
    parser.add_argument('--pos_per_bag', default=1, type=int)
    parser.add_argument('--bag_size', default=10, type=int)
    parser.add_argument('--tgt_num', default=0, type=int)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":

    torch.set_num_threads(3)
    config = get_args()
    trails = 5
    torch.manual_seed(config["seed"])

    bag_size = config["bag_size"]
    # bag_size = [5, 10, 20, 50, 100, 200, 300]
    models = ["softmax", "sparsemax", "entmax", "gsh"]    
    data_log = None

    for m in models:
        config["mode"] = m
        for t in range(trails):
            torch.random.manual_seed(torch.random.seed())
            trainer = Trainer(config, t)
            trail_log = trainer.train()
            if data_log is None:
                data_log = trail_log
            else:
                for k,v in data_log.items():
                    data_log[k] = data_log[k] + trail_log[k]
    
    sns.lineplot(data=data_log, x="epoch", y="train loss", hue="model", alpha=0.4, errorbar=None, linewidth=2)
    plt.tight_layout()
    plt.savefig(f'./imgs/cifar/train_loss_{bag_size}.pdf')
    plt.clf()

    sns.lineplot(data=data_log, x="epoch", y="test loss", hue="model", alpha=0.4, errorbar=None, linewidth=2)
    plt.tight_layout()
    plt.savefig(f'./imgs/cifar/test_loss_{bag_size}.pdf')
    plt.clf()

    sns.lineplot(data=data_log, x="epoch", y="train acc", hue="model", alpha=0.4, errorbar=None, linewidth=2)
    plt.tight_layout()
    plt.savefig(f'./imgs/cifar/train_acc_{bag_size}.pdf')
    plt.clf()

    sns.lineplot(data=data_log, x="epoch", y="test acc", hue="model", alpha=0.4, errorbar=None, linewidth=2)
    plt.tight_layout()
    plt.savefig(f'./imgs/cifar/test_acc_{bag_size}.pdf')
    plt.clf()