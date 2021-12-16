import os
import cv2
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler

import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

from matplotlib import pyplot as plt
from dataloader import *


TRAIN_DIR = '../dataset/train/'
TEST_DIR = '../dataset/test/'

# Check hardware accelerator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# Exploring Dataset
classes = os.listdir(TRAIN_DIR)
print("Total Classes: ",len(classes))

train_count = 0
test_count = 0

for _class in classes:
    train_count += len(os.listdir(TRAIN_DIR + _class))
    test_count += len(os.listdir(TEST_DIR + _class))
    
print("Total train images: ",train_count)
print("Total test images: ",test_count)


# Utility to apply transforms
def get_transform():
  mean = (127.5)
  std = (127.5)
  normalize = T.Normalize(mean=mean, std=std)
  return T.Compose([normalize])


# Loading Classification Dataset
train_dataset = CustomDataset(TRAIN_DIR, transforms=get_transform())
test_dataset = CustomDataset(TEST_DIR, transforms=get_transform())

print(len(train_dataset))
print(len(test_dataset))

train_data_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle=True)
test_data_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle=False)


def set_device():
  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"
  return torch.device(dev)


name = 'models'

try:
    os.makedirs(os.path.join(os.getcwd(), f'{name}'))
except FileExistsError:
    print("Directory already exists!")
    pass

modelDir = os.path.join(os.getcwd(), f'{name}')


# Define Training Loop

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
  device = set_device()
  train_loss = []
  train_acc = []

  for epoch in range(num_epochs):
    print("Epoch number {}".format(epoch + 1))
    start = time.time()
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
      
    # Training
    for data in train_data_loader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      total += labels.size(0)
      
      #Reset Grads
      optimizer.zero_grad()
      
      #Forward ->
      outputs = model(images)

      # pred
      _, predicted = torch.max(outputs.data, 1)

      
      #Calculate Loss & Backward, Update Weights (Step)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() 
      running_correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_data_loader)
    epoch_acc = 100.00 * running_correct / total

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    print("  - Training dataset: Got %d out of %d images correctly (%.3f%%). \nEpoch loss: %.3f"
        % (running_correct, total, epoch_acc, epoch_loss))

    test_acc = evaluate_model(model, test_loader)
      
    end = time.time()

    print("-  Epoch Time : {} \n".format(int(end-start)))

  print('Finished')
  return model, train_acc, train_loss, test_acc

# Testing function
def evaluate_model(model, test_loader):
  model.eval()
  predicted_correctly_on_epoch = 0
  total = 0
  best_acc = 0.0
  acc = []
  device = set_device()

  with torch.no_grad():
    for data in test_data_loader: 
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      total += labels.size(0)

      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      predicted_correctly_on_epoch += (predicted == labels).sum().item()
  
  epoch_acc = 100.0 * predicted_correctly_on_epoch / total
  acc.append(epoch_acc)

  if epoch_acc > best_acc:
    best_acc = epoch_acc
    torch.save(model.state_dict(), os.path.join(modelDir, 'best_model.pth'))

  print("  - Testing dataset: Got %d out of %d images correctly (%.3f%%)"
        % (predicted_correctly_on_epoch, total, epoch_acc))
  
  return acc


# Define the model
# # for resnet
# model = models.resnet18(pretrained=True)
# num_features = model.fc.in_features
# num_classes = 4
# model.fc = nn.Linear(num_features, num_classes)
# print(model)

# for mobilenet
model = models.mobilenet_v2(pretrained=True)
num_features = model.classifier[1].in_features
num_classes = 4
model.classifier[1] = nn.Linear(num_features, num_classes)
print(model)


# Define Hyperparameters
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
EPOCHS = 20


# # Training phase
model_trained, TRAIN_LOSS, TRAIN_ACC, TEST_ACC = train_model(model, train_data_loader, test_data_loader, criterion, optimizer, lr_scheduler, num_epochs=EPOCHS)

# Save the final model as well
torch.save(model.state_dict(), os.path.join(modelDir, 'model.pth'))