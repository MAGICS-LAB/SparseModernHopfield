import os
import math
import argparse

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from layers import *
from datasets.loader import load_data, DummyDataset, load_ucsb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ray import tune
from ray.air import session, RunConfig
from ray.tune.schedulers import ASHAScheduler

def get_args():

    parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
    parser.add_argument('--dataset', default='fox', type=str, choices=['fox', 'tiger', 'elephant','ucsb'])
    parser.add_argument('--mode', default='standard', type=str, choices=['standard', 'sparse'])
    parser.add_argument('--rs', help='random state', default=1111, type=int)
    parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)

    parser.add_argument('--cpus_per_trial', default=4, type=int)
    parser.add_argument('--gpus_per_trial', default=0.0, type=float)
    parser.add_argument('--gpus_id', default="0", type=str)
    args = parser.parse_args()
    return args

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_auc = 0

    def early_stop(self, validation_auc):
        if validation_auc > self.max_validation_auc:
            self.max_validation_loss = validation_auc
            self.counter = 0
        elif validation_auc < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class HopfieldMIL(nn.Module):
    def __init__(self, config, feat_dim, max_len, mode = 'standard'):
        super(HopfieldMIL, self).__init__()
        emb = [nn.Linear(feat_dim, config["emb_dims"]), nn.ReLU()]
        for i in range(config["emb_layers"] - 1):
            emb.append(nn.Linear(config["emb_dims"], config["emb_dims"]))
            emb.append(nn.ReLU())
        self.emb = nn.ModuleList(emb)
        self.mode = mode

        if mode == 'standard':
            self.hopfield = Hopfield(
                d_model=config["emb_dims"], n_heads=config["num_heads"], d_keys = config["hid_dim"],
                d_values = config["hid_dim"], scale=config["scaling_factor"],
                dropout=config["dropout"], mode='softmax'
            )
        elif mode == 'sparse':
            self.hopfield = Hopfield(
                d_model=config["emb_dims"], n_heads=config["num_heads"], d_keys = config["hid_dim"],
                d_values = config["hid_dim"], scale=config["scaling_factor"],
                dropout=config["dropout"], mode='sparsemax'
            )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config["emb_dims"], 1)
        )
        self.max_len = max_len


    def forward(self, x, mask=None):

        H = x.float()
        for l in self.emb:
            H = l(H)
        H = self.hopfield(H, stored_pattern_padding_mask=mask)
        Y_prob = self.classifier(H).flatten()

        return Y_prob

def train_epoch(network: Module,
                optimizer: torch.optim.AdamW,
                data_loader: DataLoader,
                device
               ) -> Tuple[float, float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    losses, errors, accuracies, rocs = [], [], [], []

    for data, target, mask in data_loader:
        
        data, target, mask = data.to(device=device), target.to(device=device).float(), mask.to(device)

        # Process data by Hopfield-based network.
        out = network(data, mask=mask)
        
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(input=out, target=target, reduction=r'mean')

        # Update network parameters.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        # Compute performance measures of current model.
        accuracy = (out.sigmoid().round() == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def eval_iter(network: Module,
              data_loader: DataLoader,
              device
             ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    # p_bar = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        losses, errors, accuracies, rocs, probs, labels = [], [], [], [], [], []
        for data, target, mask in data_loader:
            
            data, target, mask = data.to(device=device), target.to(device=device).float(), mask.to(device)

            # Process data by Hopfield-based network
            out = network(data, mask=mask)
            loss = F.binary_cross_entropy_with_logits(input=out, target=target, reduction=r'mean')

            # Compute performance measures of current model.
            probs = probs + (torch.sigmoid(out).squeeze(-1).tolist())
            labels = labels + (target.squeeze(-1).tolist())

            accuracy = (out.sigmoid().round() == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            roc = roc_auc_score(target.squeeze().detach().cpu(), out.sigmoid().squeeze().detach().cpu())
            rocs.append(roc)
            losses.append(loss.detach().item())

        return sum(losses) / len(losses), sum(accuracies) / len(accuracies), sum(rocs)/len(rocs)

def train(config, args, train_features, train_labels, testset):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    skf_inner = StratifiedKFold(n_splits=5, random_state=args.rs, shuffle=True)
    train_subset_ids, val_subset_ids = next(skf_inner.split(train_features, train_labels))
    train_subset_features, train_subset_labels = [train_features[id] for id in train_subset_ids] \
        , [train_labels[id] for id in train_subset_ids]
    val_subset_features, val_subset_labels = [train_features[id] for id in val_subset_ids] \
        , [train_labels[id] for id in val_subset_ids]
    train_subset, val_subset = DummyDataset(train_subset_features, train_subset_labels, args.max_len) \
        , DummyDataset(val_subset_features, val_subset_labels, args.max_len)


    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8,
        collate_fn=testset.collate
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=len(val_subset),
        shuffle=True,
        num_workers=8,
        collate_fn=testset.collate
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=len(testset),
        shuffle=False,
        num_workers=8,
        collate_fn=testset.collate
    )

    # scaling_max = 1.0
    # annealing_factor = math.pow(scaling_max/config["scaling_factor"], 1/10)
    net = HopfieldMIL(config, feat_dim=args.feat_dim, mode=args.mode, max_len=args.max_len)
    net.to(device)
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=config['lr'], weight_decay=1e-4)
    early_stopper = EarlyStopper()

    best_auc = 0.0
    for epoch in range(50):  # loop over the dataset multiple times
        epoch_steps = 0
        _ = train_epoch(net, optimizer, trainloader, device)
        # if net.mode == "sparse" and epoch%5==0:
        #     net.hopfield_pooling.hopfield.set_scaling(net.hopfield_pooling.hopfield.scaling * annealing_factor)
        epoch_steps += 1
        for g in optimizer.param_groups:
            g['lr'] *= config["lr_decay"]
        val_loss, val_acc, val_auc = eval_iter(net, valloader, device)
        if best_auc<val_auc:
            test_loss, test_acc, test_auc = eval_iter(net, testloader, device)
        if early_stopper.early_stop(val_auc):
            break
    session.report({"auc": early_stopper.max_validation_loss, "test_auc": test_auc})



def main(args, cpus_per_trial, gpus_per_trial, num_samples=1):
    features, labels = load_data(args) if args.dataset!="ucsb" else load_ucsb()
    args.feat_dim = features[0].shape[-1]
    args.max_len = max([features[id].shape[0] for id in range(len(features))])
    skf_outer = StratifiedKFold(n_splits=10, random_state=args.rs, shuffle=True)
    aucs = []
    config = {
        "lr": tune.grid_search([1e-3, 1e-5]),
        "lr_decay": tune.grid_search([0.98, 0.96, 0.94]),
        "batch_size": tune.grid_search([4]),
        "emb_dims": tune.grid_search([32, 64, 128]),
        "emb_layers": tune.grid_search([1, 2]),
        "hid_dim": tune.grid_search([16, 32, 64]),
        "num_heads": tune.grid_search([8, 12]),
        "scaling_factor": tune.grid_search([0.1, 10.0]),
        "dropout": tune.grid_search([0.0, 0.75])
    }
    for outer_iter, (train_ids, test_ids) in enumerate(skf_outer.split(features, labels)):
        train_features, train_labels = [features[id] for id in train_ids], [labels[id] for id in train_ids]
        test_features, test_labels = [features[id] for id in test_ids], [labels[id] for id in test_ids]
        testset = DummyDataset(test_features, test_labels, args.max_len)

        scheduler = ASHAScheduler(
            max_t=1,
            grace_period=1,
            reduction_factor=2)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train, args=args, train_features=train_features
                                     , train_labels=train_labels, testset=testset),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric="auc",
                mode="max",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=config,
            run_config=RunConfig(local_dir="./results"
                                 , name=f"{args.mode}_{args.dataset}_fold_{outer_iter}_rs_{args.rs}")
        )
        results = tuner.fit()

        best_result = results.get_best_result("auc", "max")

        if outer_iter==0:
            config = best_result.config

        print("Best trial final test roc-auc: {}".format(best_result.metrics["test_auc"]))
        aucs.append(best_result.metrics["test_auc"])
    print(f"dataset:{args.dataset} auc:{sum(aucs)/len(aucs)}")



if __name__ == '__main__':
    args = get_args()
    if args.gpus_per_trial>0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_id
    main(args, num_samples=1, cpus_per_trial=args.cpus_per_trial
         , gpus_per_trial=args.gpus_per_trial)


