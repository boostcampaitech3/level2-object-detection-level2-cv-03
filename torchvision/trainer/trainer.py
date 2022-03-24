import multiprocessing
import os
from importlib import import_module

from dataset.kfold import KFold
import numpy as np
import torch
import pandas as pd
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

from sklearn.metrics import f1_score

from model import criterion_entrypoint
from dataset import CustomTrainDataset
from dataset.transform import *
from utils import *


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


class Trainer:
    def __init__(self, config, annotation, data_dir, save_dir, aug_csv_path=False):
        self.annotation = annotation
        self.data_dir = data_dir
        self.save_dir = increment_path(os.path.join(save_dir, config.name))
        makedirs(self.save_dir)
        increment_name = self.save_dir.split('/')[-1]
        self.wandb = Wandb(**config.wandb, name=increment_name, config=config)
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'


    def train(self, config, pseudo_df=None):
        ''' level1 코드
        # folds = KFold(csv_path=self.csv_path, img_path=self.img_path, **config.fold)
        # train_df, val_df = folds[0] # 나중에는 fold 다 돌면서 진행해도 됨

        # -- transform
        # transform_module = getattr(import_module("dataset"), config.augmentation.name)
        # train_transform = transform_module(augment=True, **config.augmentation.args)
        # test_transform = transform_module(augment=False, **config.augmentation.args)
        
        # val_set = MaskDataset(val_df, transform=test_transform, target=config.target)
        '''

        train_dataset = CustomTrainDataset(self.annotation, self.data_dir, get_train_transform())
        
         
        train_data_loader = DataLoader(
            train_dataset,
            num_workers=multiprocessing.cpu_count()//2,
            collate_fn=collate_fn,
            pin_memory=self.use_cuda,
            **config.data_loader
        )
        '''
        # val_loader = DataLoader(
        #     val_set,
        #     num_workers=multiprocessing.cpu_count()//2,
        #     pin_memory=self.use_cuda,
        #     **config.val_data_loader
        # )'''

        print(self.device)
        
        # torchvision model 불러오기
        num_classes = 11 # class 개수= 10 + background
        model_module = getattr(import_module("model"), config.model.name)
        model = model_module(num_classes=num_classes).to(self.device)
        model = torch.nn.DataParallel(model).cuda()

        # get number of input features for the classifier

        opt_module = getattr(import_module("torch.optim"), config.optimizer.type)
        optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), **config.optimizer.args)

        ################ training #################
        best_loss = 1000
        loss_hist = Averager()

        # wandb
        self.wandb.watch(model)
        for epoch in range(config.num_epochs):
            loss_hist.reset()

            for images, targets, image_ids in tqdm(train_data_loader):

                # gpu 계산을 위해 image.to(device)
                images = list(image.float().to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # calculate loss
                loss_dict = model(config.flag, images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)

                # backward
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(
                f"Epoch #{epoch+1} / {config.num_epochs} ||"
                f"loss: {loss_hist.value:4.5} ||"
                )
            if loss_hist.value < best_loss:                
                torch.save(model.state_dict(), f"{self.save_dir}/best_loss.pth")
                best_loss = loss_hist.value
            
            print(f"best loss: {best_loss:4.5} ||")
            torch.save(model.state_dict(), f"{self.save_dir}/epoch{epoch}.pth")
            self.wandb.write_log(epoch ,loss_hist.value, best_loss)
            