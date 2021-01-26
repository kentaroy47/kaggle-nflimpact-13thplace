import os
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from pathlib import Path
from typing import List

from albumentations.pytorch.transforms import ToTensorV2
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import SupervisedRunner
from logger import WandbLogger
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

import argparse

# parsers
parser = argparse.ArgumentParser(description='NFL')
parser.add_argument('--teamcv', action='store_false')
parser.add_argument('--all', action='store_true')
parser.add_argument('--assist', action='store_true')
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--enum', default=10, type=int)
parser.add_argument('--gamma', default=3, type=int)
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--effnet', default="efficientnet_b0")
parser.add_argument('--modeltype', default="Both")
parser.add_argument('--boxaug', action='store_true', help='augment boxshapes online')
args = parser.parse_args()

fold = args.fold
teamcv = args.teamcv
mixup = args.mixup
effnet = args.effnet
assist = args.assist
modeltype = args.modeltype
boxaug = args.boxaug
enum = args.enum
gamma = args.gamma

all = args.all
savedir = "train_images_9_classification"
if all:
    savedir = "train_images_9_classification_all"
if boxaug:
    savedir = "train_images_9_classification_all_nobox"
if assist:
    savedir = "train_images_9_classification_v2"

import wandb
watermark = "classification_{}_{}".format(effnet, modeltype)
if args.teamcv:
    watermark += "_teamcv"
else:
    watermark += "_oldfolds"
if args.mixup:
    watermark += "_mixup"
if args.all:
    watermark += "_all"
if assist:
    watermark += "_assist_2frameskip"
if boxaug:
    watermark += "_boxaug"

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
BS = 100 
DATA_DIR = Path("./")
IMAGE_DIR = Path("train_images/")


class HeadClassificationDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, img_size=128, transforms=None, test=False):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.img_size = img_size
        self.test = test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        try:
            sample = self.df.loc[idx, :]
            x, y, w, h = sample.x, sample.y, sample.w, sample.h
            if not self.test:
                # aug
                x = x + int((np.random.randn()))
                y = y + int((np.random.randn()))
                w = w + int((np.random.randn(1)*2))
                h = h + int((np.random.randn(1)*2))
            
            record = self.df.loc[idx, :]
            img_id = record.image_name[:-4]+"_"+str(record.x)+"_"+str(record.y)+"_"+str(record.w)+"_"+str(record.h)+".npz"     
            cropped = (np.load(os.path.join(savedir,img_id))["arr_0"]).astype(np.float32)
            if boxaug:
                for i in range(9):
                    image = np.array(cropped[:,:,i*3:i*3+3])                    
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0))
                    cropped[:,:,i*3:i*3+3] = image
        except:
            print(img_id)
            sample = self.df.loc[100, :]
            record = self.df.loc[100, :]
            img_id = record.image_name[:-4]+"_"+str(record.x)+"_"+str(record.y)+"_"+str(record.w)+"_"+str(record.h)+".npz"     
            cropped = (np.load(os.path.join(savedir,img_id))["arr_0"]).astype(np.float32)
        cropped /= 255.0
        
        if self.transforms is not None:
            cropped = self.transforms(image=cropped)["image"]
        
        label = sample.impact - 1.0
        return {
            "image": cropped,
            "targets": np.array([label])
        }
    
    
class F1Callback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 threshold: float = 0.5,
                 prefix: str = "f1"):
        super().__init__(CallbackOrder.Metric)
        
        self.input_key = input_key
        self.output_key = output_key
        self.threshold = threshold
        self.prefix = prefix
        
    def on_loader_start(self, state: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []
            
    def on_batch_end(self, state: IRunner):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = torch.sigmoid(state.output[self.output_key].detach()).cpu().numpy()

        self.prediction.append(out)
        self.target.append(targ)

        score =  f1_score(y_true=targ, y_pred=(out > self.threshold).astype(int))
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: IRunner):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = f1_score(y_true=y_true, y_pred=(y_pred > self.threshold).astype(int))
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" + self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score
            
            
def get_train_transforms(img_size=128):
    return A.Compose([
        A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=40,shift_limit=0.2,p=0.5),
        #A.OneOf([
        #A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
        #                     val_shift_limit=0.2, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.2, 
        #                           contrast_limit=0.2, p=0.5),
        #],p=0.9),
        A.HorizontalFlip(p=0.5),
        A.Cutout(num_holes=12, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        # A.Cutout(p=0.5),
        A.Resize(height=img_size, width=img_size, p=1),
        ToTensorV2(p=1)
    ], p=1.0)


def get_valid_transforms(img_size=128):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, p=1),
        ToTensorV2(p=1.0)
    ], p=1.0)

if teamcv:
    val_videos = [
        '58005_001254_Endzone.mp4', '58005_001254_Sideline.mp4',
        '58005_001612_Endzone.mp4', '58005_001612_Sideline.mp4',
        '58048_000086_Endzone.mp4', '58048_000086_Sideline.mp4',
        '58093_001923_Endzone.mp4', '58093_001923_Sideline.mp4',
        '58094_000423_Endzone.mp4', '58094_000423_Sideline.mp4',
        '58094_002819_Endzone.mp4', '58094_002819_Sideline.mp4',
        '58095_004022_Endzone.mp4', '58095_004022_Sideline.mp4',
        '58098_001193_Endzone.mp4', '58098_001193_Sideline.mp4',
        '58102_002798_Endzone.mp4', '58102_002798_Sideline.mp4',
        '58103_003494_Endzone.mp4', '58103_003494_Sideline.mp4',
        '58104_000352_Endzone.mp4', '58104_000352_Sideline.mp4',
        '58107_004362_Endzone.mp4', '58107_004362_Sideline.mp4']
else:
    valid_videos = pd.read_csv('video_folds.csv')
    vals = valid_videos[valid_videos["fold"]==fold].video.values
    val_videos = vals

import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FocalLoss(nn.Module):
    def __init__(self, gamma=3, alpha=[0.75,0.25], size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.alpha.to(device)
        self.size_average = size_average

    def forward(self, x, t):
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        alpha = 0.25
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(x, t, w.detach(), size_average=False)
        
def main():
    video_labels = pd.read_csv(DATA_DIR / "train_labels_3.csv").fillna(0)
    if all:
        video_labels2 = pd.read_csv(DATA_DIR / "train_labels_all.csv").fillna(0)
        #video_labels2 = video_labels2[video_labels2["impact"]==1]
        video_labels2 = video_labels2[video_labels2["frame"]>1]
        video_labels2 = video_labels2[video_labels2["frame"]%6==0]
    video_labels = video_labels[video_labels["frame"]>10]
    impact_labels = video_labels[video_labels["impact"]==2]
    if all:
        video_labels = pd.concat([video_labels,impact_labels,impact_labels,impact_labels,impact_labels,impact_labels,video_labels2], axis=0)
    video_labels.reset_index(inplace=True, drop=True)
    print(video_labels.head())
    # filter labels for specific training
    if modeltype == "Both":
        # keep all labels
        pass
    else:
        # filter only end or side
        video_labels = video_labels[video_labels["view"]==modeltype]
        
    logdir = Path("./outdir"+watermark)
    logdir.mkdir(exist_ok=True, parents=True)
    set_seed(1213)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trn_df = video_labels[~video_labels.video.isin(val_videos)].reset_index(drop=True)
    val_df = video_labels[video_labels.video.isin(val_videos)].reset_index(drop=True)

    IMG_SIZE = 128
    trn_dataset = HeadClassificationDataset(trn_df, image_dir=IMAGE_DIR, img_size=IMG_SIZE, transforms=get_train_transforms(IMG_SIZE))
    val_dataset = HeadClassificationDataset(val_df, image_dir=IMAGE_DIR, img_size=IMG_SIZE, transforms=get_valid_transforms(IMG_SIZE), test=True)

    trn_loader = torchdata.DataLoader(trn_dataset, batch_size=BS, shuffle=True, num_workers=8)
    val_loader = torchdata.DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=8)

    loaders = {
        "train": trn_loader,
        "valid": val_loader
    }

    model = timm.create_model(effnet, pretrained=True)
    if effnet=="efficientnet_b0" or effnet=="efficientnet_b1" or effnet=="efficientnet_b2":
        model.conv_stem = torch.nn.Conv2d(27, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif "b3" in effnet:
        model.conv_stem = torch.nn.Conv2d(27, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif "b6" in effnet:                                      
        model.conv_stem = torch.nn.Conv2d(27, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif "se" in effnet:
        model.layer0.conv1 = torch.nn.Conv2d(27, 64, kernel_size=7,stride=2,padding=3,bias=False)
    elif "resnet" in effnet:
        model.conv1 = torch.nn.Conv2d(27, 64, kernel_size=7,stride=2,padding=3,bias=False)
    else:
        model.conv_stem = torch.nn.Conv2d(27, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
    if "eff" in effnet:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
    elif "se" in effnet:
        model.last_linear = nn.Linear(model.num_features,1)
    else:
        model.fc = nn.Linear(model.num_features,1)
    model.to(device);
    print(model)

    criterion = FocalLoss(gamma)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=enum)
    callbacks = [
        F1Callback(input_key="targets",
                   threshold=0.5,
                   prefix="f1_at_05"),
        F1Callback(input_key="targets",
                   threshold=0.7,
                   prefix="f1_at_07"),
        F1Callback(threshold=0.3,
                   prefix="f1_at_03"),
        WandbLogger(project="nfl_classification2",name= watermark)]

    runner = SupervisedRunner(device=device,
                              input_key="image",
                              input_target_key="targets")

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=enum,
        verbose=True,
        logdir=logdir,
        callbacks=callbacks,
        main_metric="epoch_f1_at_03",
        minimize_metric=False)
    

if __name__ == "__main__":
    main()
