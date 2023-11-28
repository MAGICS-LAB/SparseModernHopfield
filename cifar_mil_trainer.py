from datasets.cifar10_bags import CIFARBags
from torch.utils.data import DataLoader
from layers import *
from models import *
import wandb
import pandas as pd

class Trainer:

    def __init__(self, config, trial) -> None:
        self.config = config
        self.trial = trial

        if self.config["wandb"]:
            run = wandb.init(
                # Set the project where this run will be logged
                project=self.config["project_name"] + " good",
                # Track hyperparameters and run metadata
                config=self.config)
    
    def _get_data(self):

        trainset = CIFARBags(target_number=self.config["tgt_num"], 
                                  bag_size=self.config["bag_size"], 
                                  num_bag=self.config["train_size"],
                                  pos_per_bag=self.config["pos_per_bag"],
                                  seed=self.config["seed"],
                                  train=True
                                  )  

        testset = CIFARBags(target_number=self.config["tgt_num"], 
                                  bag_size=self.config["bag_size"], 
                                  num_bag=self.config["test_size"],
                                  pos_per_bag=self.config["pos_per_bag"],
                                  seed=self.config["seed"],
                                  train=False
                                  )  

        train_loader = DataLoader(trainset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(testset, batch_size=self.config["batch_size"], shuffle=False)

        return train_loader, test_loader

    def _get_model(self):

        model = MNISTModel(input_size=self.config["input_size"],
                            d_model=self.config["d_model"],
                            n_heads=self.config["n_heads"], 
                            update_steps=self.config["update_steps"], 
                            dropout=self.config["dropout"],
                            mode=self.config["mode"],
                            scale=self.config["scale"],
                            num_pattern=self.config['num_pattern'])

        return model.cuda()

    def _get_opt(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config["lr"], weight_decay=0.0)

    def _get_cri(self):
        return torch.nn.BCEWithLogitsLoss()

    def test_epoch(self, loader):

        total_loss = 0.0
        total_cor, total_sample = 0, 0
        total_step = 0

        with torch.no_grad():
            for x, y in loader:

                total_sample += x.size(0)
                total_step += 1
                x, y = x.float().cuda(), y.float().cuda()
                pred = self.model(x)
                loss = self.cri(pred, y)

                output = (pred>0.5).float()
                total_cor += (output == y).float().sum()
                total_loss += loss.item()
        
        return total_loss/total_step, total_cor/total_sample

    def train_epoch(self, loader):

        total_loss = 0.0
        total_cor, total_sample = 0, 0
        total_step = 0

        for x, y in loader:

            total_step += 1
            total_sample += x.size(0)

            self.opt.zero_grad()
            x, y = x.float().cuda(), y.float().cuda()
            pred = self.model(x)
            loss = self.cri(pred, y)
            loss.backward()
            self.opt.step()

            output = (pred>0.5).float()
            total_cor += (output == y).float().sum()
            total_loss += loss.item()
        
        return total_loss/total_step, total_cor/total_sample

    def train(self):

        train_loader, test_loader = self._get_data()
        self.model = self._get_model()
        self.opt = self._get_opt()
        self.cri = self._get_cri()

        best_test_acc = -1

        data_log = {
            'train loss':[],
            'train acc':[],
            'test loss':[],
            'test acc':[],
            'epoch':[],
            'model':[]
        }

        self.sche = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.config["epoch"], eta_min=0, last_epoch=-1, verbose=False)

        for epoch in range(1, self.config["epoch"]+1):

            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.test_epoch(test_loader)
            self.sche.step()

            data_log['train loss'].append(train_loss)
            data_log['test loss'].append(test_loss)
            data_log['train acc'].append(train_acc.item())
            data_log['test acc'].append(test_acc.item())
            data_log['epoch'].append(epoch)
            data_log['model'].append(self.config['mode'])

            if test_acc >= best_test_acc:
                best_test_acc = test_acc

            if self.config["wandb"]:
                wandb.log({
                    "step": epoch,
                    "train loss": train_loss,
                    "train acc": train_acc.item()*100,
                    "test loss": test_loss,
                    "test acc": test_acc.item()*100
                    }, step=epoch)
                
        if self.config["wandb"]:
            wandb.log({"best test acc": best_test_acc})
            wandb.log({"logs": data_log})
        
        if self.config["wandb"]:
            wandb.finish()

        return data_log
