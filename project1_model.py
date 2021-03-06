# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LAG4l4ngRlyP-WLWt6g-1p65q56MBJBm
"""

import time
import sys
import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models

Root = ".data"
transform = transforms.Compose([transforms.Resize((32,32)), 
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10), 
                                transforms.RandomCrop(size=32, padding=[0, 2, 3, 4]), 
                                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), 
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transformation=transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
train = torchvision.datasets.CIFAR10(Root, 
                                     train= True,
                                     transform= transform,
                                     download= True)
test = torchvision.datasets.CIFAR10(Root, 
                                     train= False,
                                     transform= transformation,  
                                     download= True)

batch= 128
trainloader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
testloader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=2)
images, labels = iter(trainloader).next()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 40

        self.conv1 = nn.Conv2d(3, 40, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.layer1 = self._make_layer(block, 40, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 80, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 160, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 320, num_blocks[3], stride=2)
        self.linear = nn.Linear(320, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def res():
      return ResNet(BasicBlock, [2, 2, 2, 2])

project1_model = res()
print(project1_model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_of =  count_parameters(project1_model)
print("Count of Parameters: ",num_of)

#Defining loss fn and Optimiser
device = "cuda"

optimizer = torch.optim.Adam(project1_model.parameters(), lr = 0.001)

CE_loss= nn.CrossEntropyLoss().to(device)

## Defining the Training Loop
def training_n_testing(trainloader, testloader, model, loss_fn, optimiser):
    running_loss=0
    correct=0
    total=0
    
### Training Section ############################################    
    for batch, (imgs, labels) in enumerate(trainloader):

        X, y = imgs.to(device), labels.to(device)
        pred = model(X)

        loss = loss_fn(pred, y)

        # Back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()
        
    train_loss = running_loss/len(trainloader)
    train_loss_all.append(train_loss)
    train_accuracy=100.*correct/total
    train_accuracy_all.append(train_accuracy)

### Testing Section ###########################################################
    running_loss=0
    correct=0
    total=0
    test_model = copy.deepcopy(model)
        
    for batch, (imgs, labels) in enumerate(testloader):
        X = imgs.to(device)
        y= labels.to(device)
        pred = test_model(X)
        loss = loss_fn(pred, y)

        running_loss += loss.item()
        _, predicted = pred.max(1)
        total += labels.size(0)  
        correct += predicted.eq(labels.to(device)).sum().item()


    test_loss = running_loss/len(testloader)
    test_loss_all.append(test_loss)
    test_accuracy = 100.*correct/total
    test_accuracy_all.append(test_accuracy)

    print('Epoch_number:%d |Train Loss: %.3f| Test_loss: %.3f| Train Accu: %.3f| Test Accu: %.3f'%(epoch, train_loss, test_loss, train_accuracy, test_accuracy))

## Finally Start Training
device ="cuda"
train_loss_all = [] 
test_loss_all = [] 
train_accuracy_all = [] 
test_accuracy_all = []
project1_model= res()
project1_model.to(device)
optimizer = torch.optim.Adam(project1_model.parameters(), lr = 0.001)
CE_loss= nn.CrossEntropyLoss().to(device)

epochs = 120
for epoch in range(epochs):
    
    training_n_testing(trainloader, testloader, project1_model, CE_loss, optimizer)

    if (epochs>90 and epochs<110):
      lr=0.01
    elif (epochs>110):
      lr=0.1

print("Done!")

### Plotting the Accuracy and Losses for Training and Testing
plt.plot(range(epochs), train_loss_all, 'b', label='Training loss')
plt.plot(range(epochs), test_loss_all, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

### Plotting the Accuracy and Losses for Training and Testing
plt.plot(range(epochs), train_accuracy_all, 'b', label='Training Accu')
plt.plot(range(epochs), test_accuracy_all, 'r', label='Test Accu')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

train_accuracy_all = np.array(train_accuracy_all)
test_accuracy_all = np.array(test_accuracy_all)
train_loss_all = np.array(train_loss_all)
test_loss_all = np.array(test_loss_all)

np.save('train_accuracy7.npy', train_accuracy_all)
np.save('test_accuracy7.npy', test_accuracy_all)
np.save("train_loss7.npy", train_loss_all)
np.save("test_loss7.npy", test_loss_all)






