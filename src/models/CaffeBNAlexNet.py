import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class CaffeBNAlexNet(nn.Module):
  def __init__(self, num_classes=10):
    super(CaffeBNAlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(96, 256, kernel_size=5, padding=2),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(256, 384, kernel_size=3, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 384, kernel_size=3, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 256, kernel_size=3, padding=0),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Linear(256 * 6 * 6, 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, num_classes),
    )
    self.loss = lambda x, target: nn.CrossEntropyLoss()(x, target)


  def forward(self, x):
    x = self.features(x)
    #print(x.shape)
    x = torch.flatten(x, 1)
    #print(x.shape)
    x = self.classifier(x)
    return x
