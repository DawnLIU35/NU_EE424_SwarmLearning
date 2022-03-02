from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

import os


def split(df,p=0.2):
    clipNames = sorted(df['clipNames'].unique())

    dayClips = [name for name in clipNames if 'day' in name]

    testDayClipNames = list(np.random.choice(dayClips,int(len(dayClips)*p)))
    testClipNames =  testDayClipNames

    trainDayClipNames = list(set(dayClips) - set(testDayClipNames))
    trainClipNames =  trainDayClipNames
    
    train_df = df[df.clipNames.isin(trainClipNames)]
    test_df = df[df.clipNames.isin(testClipNames)]
    
    return train_df, test_df

class TrafficLightsDataset:
    def __init__(self, df, transforms=None):
        super().__init__()

        self.image_ids = df.image_id.unique()
        self.df = df
        self.transforms = transforms
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df.image_id == image_id]

        image = cv2.imread(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        motion_labels = torch.as_tensor(self.df.label.values[index],dtype=torch.float32)
        
        boxes = records[['x_min','y_min','x_max','y_max']].values
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.as_tensor(records.label.values, dtype=torch.int64)
        
        iscrowd = torch.zeros_like(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.as_tensor(sample['bboxes'],dtype=torch.float32)
            target['labels'] = torch.as_tensor(sample['labels'])
            
        return image, motion_labels #, target, image_id

def getTrainTransform():
    return A.Compose([
        A.Resize(height=256, width=256, p=1),
        # A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def getValTransform():
    return A.Compose([
        A.Resize(height=256, width=256, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def getTestTransform():
    return A.Compose([
        A.Resize(height=256, width=256, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def GetDataset(args):
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    device = args.device
    DATA_PATH = args.parent_path
    DAY_TRAIN_PATH = args.annotation_path
    np.random.seed(args.seed)
    # NIGHT_TRAIN_PATH = './RawData/Annotations/nightTrain/'
    train_day = []
    for clipName in tqdm(sorted(os.listdir(DAY_TRAIN_PATH))):
        df = pd.read_csv(os.path.join(DAY_TRAIN_PATH,clipName,'frameAnnotationsBOX.csv'),sep=';')
        train_day.append(df)
    train_day_df = pd.concat(train_day,axis=0)

    df = train_day_df.drop(['Origin file','Origin track','Origin track frame number'],axis=1)

    def changeFilename(x):
        filename = x.Filename
        splitted = filename.split('/')
        clipName = splitted[-1].split('--')[0]
        if args.image_time == 'day':
            return os.path.join(DATA_PATH,f'dayTrain/dayTrain/{clipName}/frames/{splitted[-1]}')
        elif args.image_time == 'night':
            return os.path.join(DATA_PATH,f'nightTrain/nightTrain/{clipName}/frames/{splitted[-1]}')
    
    df['Filename'] = df.apply(changeFilename,axis=1)

    label_to_idx = {'go':0, 'warning':1, 'stop': 2}
    idx_to_label = {v:k for k,v in label_to_idx.items()}

    def changeAnnotation(x):
        if 'go' in x['Annotation tag']:
            return label_to_idx['go']
        elif 'warning' in x['Annotation tag']:
            return label_to_idx['warning']
        elif 'stop' in x['Annotation tag']:
            return label_to_idx['stop']
        
    df['Annotation tag'] = df.apply(changeAnnotation,axis=1)

    annotation_tags = df['Annotation tag'].unique()

    df.columns = ['image_id','label','x_min','y_min','x_max','y_max','frame']

    df['clipNames'] = df[['image_id']].applymap(lambda x: x.split('/')[4])

    train_df, test_df = split(df)
    train_df, val_df = split(train_df)

    trainDataset = TrafficLightsDataset(train_df,getTrainTransform())
    valDataset = TrafficLightsDataset(val_df,getValTransform())
    testDataset = TrafficLightsDataset(test_df,getTestTransform())

    return trainDataset, valDataset, testDataset