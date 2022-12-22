import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from collections import OrderedDict
import numpy as np

import sys
sys.path.append("..")

from utils import generate_low_rank_conv2d, generate_low_rank_linear

class CaffeBNLowRankAlexNet(nn.Module):
  def __init__(self, ranks, scheme, num_classes=10):
    super(CaffeBNLowRankAlexNet, self).__init__()
    self.features = nn.Sequential(
      generate_low_rank_conv2d(3, 96, kernel_size=11, stride=4, padding=2, rank=ranks[0], scheme=scheme),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      generate_low_rank_conv2d(96, 256, kernel_size=5, padding=2, rank=ranks[1], scheme=scheme),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      generate_low_rank_conv2d(256, 384, kernel_size=3, padding=1, rank=ranks[2], scheme=scheme),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      generate_low_rank_conv2d(384, 384, kernel_size=3, padding=1, rank=ranks[3], scheme=scheme),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      generate_low_rank_conv2d(384, 256, kernel_size=3, padding=0, rank=ranks[4], scheme=scheme),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      generate_low_rank_linear(256 * 6 * 6, 4096, rank=ranks[5]),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      generate_low_rank_linear(4096, 4096, rank=ranks[6]),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      generate_low_rank_linear(4096, num_classes, rank=ranks[7]),
    )
    self.loss = lambda x, target: nn.CrossEntropyLoss()(x, target)

def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
