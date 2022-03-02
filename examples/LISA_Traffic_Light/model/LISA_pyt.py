############################################################################
## Copyright 2021 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
############################################################################

import datetime
import numpy as np
import os
from swarm import SwarmCallback
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pdb
import math
from tqdm import tqdm

from time import time
import pandas as pd

import seaborn as sns

import cv2

from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision

from torchvision.ops import nms

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
import os

class arguments():
    split = 'random'   # random: randomly split train and test in all images. clip: randomly split based on clips
    samples = 0.1 # only use partial of dataset.
    pixel = 256 # image pixels
    epochs = 5
    lr = 0.0001
    batch_size = 64
    seed = 0
    device = 'cpu'
    image_time = 'day' # or night
    parent_path = '../app-data/RawData/'
    annotation_path = '../app-data/RawData/Annotations/'+image_time+'Train'   # './RawData/Annotations/nightTrain/'
args = arguments

default_max_epochs = args.epochs
default_min_peers = 2
trainPrint = True
# tell swarm after how many batches
# should it Sync. We are not doing 
# adaptiveRV here, its a simple and quick demo run
swSyncInterval = 128 

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, 3)
        
    def forward(self, x):  
        x = self.resnet(x)
        # output = F.log_softmax(x, dim=1)
        return x
        
def loadData(args):
    trainDataset, valDataset, testDataset = GetDataset(args)
    return trainDataset, testDataset

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


def GetDataset(args):
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    device = args.device
    DATA_PATH = args.parent_path
    DAY_TRAIN_PATH = args.annotation_path
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # NIGHT_TRAIN_PATH = './RawData/Annotations/nightTrain/'
    train_day = []
    for clipName in sorted(os.listdir(DAY_TRAIN_PATH)):
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

    df['clipNames'] = df[['image_id']].applymap(lambda x: x.split('/')[5])

    def getTrainTransform():
        return A.Compose([
            A.Resize(height=args.pixel, width=args.pixel, p=1),
            # A.Flip(0.5),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    def getValTransform():
        return A.Compose([
            A.Resize(height=args.pixel, width=args.pixel, p=1),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    def getTestTransform():
        return A.Compose([
            A.Resize(height=args.pixel, width=args.pixel, p=1),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    # two ways to split dataset
    if args.split == 'clips':
        train_df, test_df = split(df)
        train_df, val_df = split(train_df)

        trainDataset = TrafficLightsDataset(train_df,getTrainTransform())
        valDataset = TrafficLightsDataset(val_df,getValTransform())
        testDataset = TrafficLightsDataset(test_df,getTestTransform())
    
    elif args.split == 'random':
        ds = TrafficLightsDataset(df, getTrainTransform())
        trainDataset, testDataset = torch.utils.data.random_split(ds, [len(ds)-math.ceil(0.2*len(ds)), math.ceil(0.2*len(ds))])
        valDataset = 0

    return trainDataset, valDataset, testDataset

def doTrainBatch(model,device,trainLoader,optimizer,epoch,swarmCallback):
    model.train()
    for batchIdx, (data, target) in enumerate(tqdm(trainLoader)):
        data, target = torch.stack(data).to(device), torch.stack(target).type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if trainPrint and batchIdx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batchIdx * len(data), len(trainLoader.dataset),
                  100. * batchIdx / len(trainLoader), loss.item()))
        # Swarm Learning Interface
        if swarmCallback is not None:
            swarmCallback.on_batch_end()        

def test(model, device, testLoader):
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = torch.stack(data).to(device), torch.stack(target).type(torch.LongTensor).to(device)
            output = model(data)
            # testLoss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            loss_fn = nn.CrossEntropyLoss()
            testLoss += loss_fn(output, target).item()  # sum up batch loss
            output_prob = F.softmax(output, dim=1)
            pred = output_prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(testLoader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        testLoss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))    

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    args = arguments
    torch.manual_seed(args.seed)
    modelDir = os.getenv('MODEL_DIR', './model')
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    batchSz = args.batch_size # this gives 97% accuracy on CPU

    # dataset
    trainDs, testDs = loadData(args)
    trainDs, unDS = torch.utils.data.random_split(trainDs, [math.ceil(len(trainDs)*args.samples),len(trainDs)-math.ceil(len(trainDs)*args.samples)])
    testDs, unDS = torch.utils.data.random_split(testDs, [math.ceil(len(testDs)*args.samples),len(testDs)-math.ceil(len(testDs)*args.samples)])

    useCuda = torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")  
    model = ResNet().to(device)
    model_name = 'LISA_pyt'
    opt = optim.Adam(model.parameters(),lr=args.lr)

    trainLoader = torch.utils.data.DataLoader(
        trainDs,
        batch_size=batchSz,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
        )

    testLoader = torch.utils.data.DataLoader(
            testDs,
            batch_size=batchSz,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
    print('train data size: {}, test data size: {}'.format(len(trainLoader.dataset), len(testLoader.dataset)))

    # Create Swarm callback
    swarmCallback = None
    swarmCallback = SwarmCallback(sync_interval=swSyncInterval,
                                  min_peers=min_peers,
                                  val_data=testDs,
                                  val_batch_size=batchSz,
                                  model_name=model_name,
                                  model=model)
    # initalize swarmCallback and do first sync 
    swarmCallback.on_train_begin()
        
    for epoch in range(1, max_epochs + 1):
        doTrainBatch(model,device,trainLoader,opt,epoch,swarmCallback)      
        test(model,device,testLoader)
        swarmCallback.on_epoch_end(epoch)

    # handles what to do when training ends        
    swarmCallback.on_train_end()

    # Save model and weights
    model_path = os.path.join(modelDir, model_name, 'saved_model.pt')
    # Pytorch model save function expects the directory to be created before hand.
    os.makedirs(os.path.join(modelDir, model_name), exist_ok=True)
    torch.save(model, model_path)
    print('Saved the trained model!')
  
if __name__ == '__main__':
  main()
