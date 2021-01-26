# training script for NFL
# arutema47

import argparse
# parsers
parser = argparse.ArgumentParser(description='NFL')
parser.add_argument('--imsize', default=768, type=int)
parser.add_argument('--enum', default=25, type=int)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--strech', default=4, type=int)
parser.add_argument('--eval', action='store_true', help='add mixup augumentations')
parser.add_argument('--mixup', action='store_false', help='add mixup augumentations')
parser.add_argument('--cutmix', action='store_true', help='add mixup augumentations')
parser.add_argument('--modeltype', default="Endzone")
parser.add_argument('--effdet', default="effdet5")
parser.add_argument('--cont', action='store_true')
parser.add_argument('--clip', action='store_false')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--clean', action='store_false')
parser.add_argument('--all', action='store_true')
parser.add_argument('--after', action='store_true')
parser.add_argument('--teamcv', action='store_false')
parser.add_argument('--oversample', action='store_true')
parser.add_argument('--nowandb', action='store_true')
parser.add_argument('--saveall', action='store_true')
parser.add_argument('--modelpath', default="last-checkpoint.bin", type=str)
parser.add_argument('--impactonly', action='store_true', help='add mixup augumentations')

args = parser.parse_args()
fold = args.fold

import sys
sys.path.insert(0, "effdet2/")
import effdet

import copy
import torch
import torch.nn as nn
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import pandas as pd
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from tqdm import tqdm

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# training options
IMSIZE = args.imsize
ENUM = args.enum
BSNUM = args.bs
LR = args.lr
evalontrain = args.eval
debug = False
cutmix = args.cutmix
mixup = args.mixup
CLIP = args.clip
modeltype = args.modeltype
impactonly = args.impactonly
strech = args.strech
effdet_arch = args.effdet
OVERSAMPLE = args.oversample

# other options
CONTINUE = args.cont
MODELPATH = args.modelpath
use_amp = args.amp
all = args.all
after = args.after
clean = args.clean
team_cv = args.teamcv
saveall = args.saveall

watermark = "{}-{}".format(effdet_arch, IMSIZE)+"-"+modeltype+"-fold"+str(fold)+"-strech{}".format(strech)
if cutmix:
    watermark += "_cutmix"
if mixup:
    watermark += "_mixup"
if impactonly:
    watermark += "_impactonly"
if all:
    watermark += "_all"
if after:
    watermark += "_after"
if OVERSAMPLE:
    watermark += "_oversample"
if clean:
    watermark += "_clean"
if team_cv:
    watermark += "_teamcv"
if saveall:
    watermark += "_saveall"

# In[3]:
print("watermark:", watermark)
if not args.nowandb:
    import wandb
    wandb.init(project="nfl-impact-3",
               name=watermark)
LOAD_DIR = watermark

# ## Albumentations

# In[4]:


def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5, rotate_limit=30, shift_limit=0.2,
                  scale_limit=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                             val_shift_limit=0.2, p=0.75),
                        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                                   contrast_limit=0.2, p=0.75),]),            
            #A.RandomCrop(height=int(600), width=int(1024), p=0.25),              
            A.Cutout(num_holes=64, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
            A.MotionBlur(blur_limit=20,p=0.5),           
            A.Resize(height=IMSIZE, width=IMSIZE, p=1),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=IMSIZE, width=IMSIZE, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )


# ## Dataset
# read labels
video_labels = pd.read_csv('train_labels_{}.csv'.format(strech)).fillna(0)
                           
# impact only?
if impactonly:
    video_labels = video_labels[video_labels["impact"]==2]
                           
# enlarge trainsets?
if all:
    video_labels2 = pd.read_csv("train_labels_all.csv")
    video_labels2 = video_labels2[video_labels2["impact"]==1]
    print(len(video_labels2))
    if after:
        video_labels2 = video_labels2[video_labels2["frame"]%4==0]
    else:
        video_labels2 = video_labels2[video_labels2["frame"]%2==0]
    video_labels2["all"] = 1
    video_labels["all"] = 0
    print(len(video_labels2))
    orig_labels = video_labels
    video_labels = pd.concat([video_labels, video_labels2], axis=0)
    video_labels.reset_index(inplace=True)

# load hold-out settings from separate file
if not team_cv:
    valid_videos = pd.read_csv('video_folds.csv')
    vals = valid_videos[valid_videos["fold"]==fold].video.values
    video_valid = vals
else:
    video_valid = np.array(['58005_001254_Endzone.mp4', '58005_001254_Sideline.mp4',
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
                        '58107_004362_Endzone.mp4', '58107_004362_Sideline.mp4'])
    vals = video_valid
print("val videos:", vals)
                           
# filter labels for specific training
if modeltype == "Both":
    # keep all labels
    pass
else:
    # filter only end or side
    video_labels = video_labels[video_labels["view"]==modeltype]
    orig_labels = orig_labels[orig_labels["view"]==modeltype]
                           
# split to train and val labels
vect = []
vs = video_labels.video.values
for v in vs:
    if v in vals:
        vect.append(True)
    else:
        vect.append(False)
vect = np.array(vect)
val_labels = video_labels[vect]
if all:
    val_labels = val_labels[val_labels["all"]==0] # filter non-impact images
train_labels = video_labels[~vect]
train_labels = train_labels[train_labels["frame"]>0] # filter 0-frames
if all:
    impact_labels = train_labels[train_labels["all"]==0] 
    vect = []
    vs = orig_labels.video.values
    for v in vs:
        if v in vals:
            vect.append(True)
        else:
            vect.append(False)
    vect = np.array(vect)
    orig_labels = orig_labels[~vect]
else:
    impact_labels = train_labels

# Train on kouhan images?
if after:
    train_labels = train_labels[train_labels["frame"]>100] # filter 0-frames
    val_labels = val_labels[(val_labels["frame"]>100)] # filter 0-frames

# generate images for training..
images_valid = val_labels.image_name.unique()
images_train = train_labels.image_name.unique()
if all:
    tuika = orig_labels.image_name.unique()
    images_train = np.concatenate([images_train, tuika, tuika], axis=0)
images_impact = impact_labels.image_name.unique()

print("train:{}, val:{}".format(len(images_train), len(images_valid)))

print(video_labels.head())
print(video_labels.tail())
print(video_valid)
    
TRAIN_ROOT_PATH = 'train_images'

########### Set Dataset ###########
class DatasetRetriever(Dataset):
    def __init__(self, marking, image_ids, transforms=None, test=False, impactimg=None):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        self.impactimg = impactimg

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        ok = 0
        while not ok:
            rd = random.random()
            if self.test or rd < 0.5:
                image, boxes, labels = self.load_image_and_boxes(index)
                cut = 0
            elif mixup and rd < 0.8:
                image, boxes, labels = self.load_mixup_image_and_boxes(index)
                cut = 0
            elif cutmix:
                image, boxes, labels = self.load_cutmix_image_and_boxes(index)
                cut = 1
            else:
                image, boxes, labels = self.load_image_and_boxes(index)
                cut = 0

            target = {}
            target['boxes'] = boxes
            target['labels'] = torch.tensor(labels)
            target['image_id'] = torch.tensor([index])

            if self.transforms:
                for i in range(10):
                    sample = self.transforms(**{
                        'image': image,
                        'bboxes': target['boxes'],
                        'labels': labels
                    })
                    if len(sample['bboxes']) > 0:
                        image = sample['image']
                        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                        target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                        ok = 1
                        break
                        
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        #print(f'{TRAIN_ROOT_PATH}/{image_id}')
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_name'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['impact'].values
        return image, boxes, labels
    
    def load_impact_image_and_boxes(self, index):
        image_id = self.impactimg[index]
        #print(f'{TRAIN_ROOT_PATH}/{image_id}')
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_name'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['impact'].values
        return image, boxes, labels
    
    def load_mixup_image_and_boxes(self, index):
        image, boxes, labels = self.load_image_and_boxes(index)
        if not OVERSAMPLE:
            r_image, r_boxes, r_labels = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        else:
            r_image, r_boxes, r_labels = self.load_impact_image_and_boxes(random.randint(0, self.impactimg.shape[0] - 1))
        #print(r_labels)
        #print(labels)
        return (image+r_image)/2, np.vstack((boxes, r_boxes)).astype(np.int32), np.concatenate((labels, r_labels)).astype(np.int32)
    
    def load_cutmix_image_and_boxes(self, index, imsize=1280):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = 1280, 720
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        if not OVERSAMPLE:
            indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]
        else:
            indexes = [index] + [random.randint(0, self.impactimg.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            if i==0 or not OVERSAMPLE:
                image, boxes, labels = self.load_image_and_boxes(index)
            else:
                image, boxes, labels = self.load_impact_image_and_boxes(index)
                
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_labels = result_labels[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes, result_labels


# In[9]:


train_dataset = DatasetRetriever(
    image_ids=images_train,
    marking=video_labels,
    transforms=get_train_transforms(),
    test=False,
    impactimg=images_impact
)

validation_dataset = DatasetRetriever(
    image_ids=images_valid,
    marking=video_labels,
    transforms=get_valid_transforms(),
    test=True,
)

########### Set Monitor ###########
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import warnings

warnings.filterwarnings("ignore")
########### Set Fitter ###########
class Fitter:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def fit(self, train_loader, validation_loader):
        if CONTINUE:
            #try:
                self.load(f'{LOAD_DIR}/{MODELPATH}')
            #except:
            #    print("prev not found")
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            
            summary_losstrain = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_losstrain.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)
            
            # Log training..
            if not args.nowandb:
                try:
                    wandb.log({'epoch': e, 'train_loss': summary_losstrain.avg, 'val_loss': summary_loss.avg, "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": time.time()-t})
                except:
                    wandb.log({'epoch': e, 'train_loss': summary_loss.avg, 'val_loss': summary_loss.avg, "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": time.time()-t})

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)
            
            if saveall:
                self.save(f'{self.base_dir}/{self.base_dir}_epoch_{e}.bin')
            
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]
                with torch.cuda.amp.autocast():
                    loss, _, _ = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        if evalontrain:
            self.model.eval()
        else:
            self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, _, _ = self.model(images, boxes, labels)
            
            self.scaler.scale(loss).backward()
            
            #Gradient Value Clipping
            if CLIP:
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.5)
            
            summary_loss.update(loss.detach().item(), batch_size)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if self.config.step_scheduler:
                self.scheduler.step()
                
            if debug and step==10:
                break

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


# In[13]:


class TrainGlobalConfig:
    num_workers = 8
    batch_size = BSNUM
    n_epochs = ENUM
    lr = LR
    folder = watermark
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=2,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )


# In[14]:


def collate_fn(batch):
    return tuple(zip(*batch))

def run_training():
    device = torch.device('cuda:0')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)


# In[15]:


from effdet import get_efficientdet_config
from effdet.efficientdet import HeadNet

def load_net(effdet_arch):
    if effdet_arch=="effdet5":
        config = get_efficientdet_config('tf_efficientdet_d5')
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load('pretrained/efficientdet_d5-ef44aea8.pth')
        net.load_state_dict(checkpoint)
        config = get_efficientdet_config('tf_efficientdet_d5')
    elif effdet_arch=="effdet6":
        config = get_efficientdet_config('tf_efficientdet_d6')
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load('pretrained/efficientdet_d6-51cb0132.pth')
        net.load_state_dict(checkpoint)
        config = get_efficientdet_config('tf_efficientdet_d6')
    elif effdet_arch=="effdet4":
        config = get_efficientdet_config('tf_efficientdet_d4')
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load('pretrained/efficientdet_d4-5b370b7a.pth')
        net.load_state_dict(checkpoint)
        config = get_efficientdet_config('tf_efficientdet_d4')
    elif effdet_arch=="effdet3":
        config = get_efficientdet_config('tf_efficientdet_d3')
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load('pretrained/efficientdet_d3-b0ea2cbc.pth')
        net.load_state_dict(checkpoint)
        config = get_efficientdet_config('tf_efficientdet_d3')
    config.num_classes = 2
    config.image_size = IMSIZE
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return DetBenchTrain(net, config)

net = load_net(effdet_arch)

# In[16]:


run_training()


# In[ ]:


# TODO mAP for impact class?

