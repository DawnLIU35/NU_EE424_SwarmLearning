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
import timeit
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
    parent_path = os.getenv('DATA_DIR', './data') + '/RawData/'
    annotation_path = os.getenv('DATA_DIR', './data')+'/RawData/Annotations/'+image_time+'Train'  # './RawData/Annotations/nightTrain/'
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

def loadData(dataDir):
    # load data from npz format to numpy 
    path = os.path.join(dataDir,'TrafficImages.npz')
    with np.load(path) as f:
        xTrain, yTrain = f['x_train'], f['y_train']
        xTest, yTest = f['x_test'], f['y_test']
        xTrain, xTest = xTrain / 255.0, xTest / 255.0        
        
    # transform numpy to torch.Tensor
    xTrain, yTrain, xTest, yTest = map(torch.tensor, (xTrain.astype(np.float32), 
                                                      yTrain.astype(np.int_), 
                                                      xTest.astype(np.float32),
                                                      yTest.astype(np.int_)))    
    # convert torch.Tensor to a dataset
    yTrain = yTrain.type(torch.LongTensor)
    yTest = yTest.type(torch.LongTensor)
    trainDs = torch.utils.data.TensorDataset(xTrain,yTrain)
    testDs = torch.utils.data.TensorDataset(xTest,yTest)
    return trainDs, testDs

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

def main():
    args = arguments
    dataDir = os.getenv('DATA_DIR', './data')
    torch.manual_seed(args.seed)
    modelDir = os.getenv('MODEL_DIR', './model')
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    batchSz = args.batch_size # this gives 97% accuracy on CPU

    # dataset
    trainDs, testDs = loadData(dataDir)
    trainDs, unDS = torch.utils.data.random_split(trainDs, [math.ceil(len(trainDs)*args.samples),len(trainDs)-math.ceil(len(trainDs)*args.samples)])
    testDs, unDS = torch.utils.data.random_split(testDs, [math.ceil(len(testDs)*args.samples),len(testDs)-math.ceil(len(testDs)*args.samples)])

    useCuda = torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")  
    model = ResNet().to(device)
    model_name = 'LISA_pyt'
    opt = optim.Adam(model.parameters(),lr=args.lr)
    trainLoader = torch.utils.data.DataLoader(trainDs,batch_size=batchSz)
    testLoader = torch.utils.data.DataLoader(testDs,batch_size=batchSz)

    print('########## train data size: {}, test data size: {}'.format(len(trainLoader.dataset), len(testLoader.dataset)))

    # Create Swarm callback
    swarmCallback = None
    swarmCallback = SwarmCallback(sync_interval=swSyncInterval,
                                  min_peers=min_peers,
                                  val_data=testDs,
                                  val_batch_size=batchSz,
                                  model_name=model_name,
                                  model=model)
    # initalize swarmCallback and do first sync 
    print('#### begin on train begin')
    swarmCallback.on_train_begin()
    print('########## Created Swarm callback ##############')
    
    start = timeit.default_timer() 
    for epoch in range(1, max_epochs + 1):
        doTrainBatch(model,device,trainLoader,opt,epoch,swarmCallback)    
        print('######### finish training epoch {}! ##########'.format(epoch))  
        test(model,device,testLoader)
        swarmCallback.on_epoch_end(epoch)

    
    # handles what to do when training ends        
    swarmCallback.on_train_end()
    stop = timeit.default_timer()
    print('Training Used Time: ', stop - start) 
    
    # Save model and weights
    model_path = os.path.join(modelDir, model_name, 'saved_model.pt')
    # Pytorch model save function expects the directory to be created before hand.
    os.makedirs(os.path.join(modelDir, model_name), exist_ok=True)
    torch.save(model, model_path)
    print('Saved the trained model!')
  
if __name__ == '__main__':
  main()
