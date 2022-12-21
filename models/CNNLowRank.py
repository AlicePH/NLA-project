import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import generate_low_rank_conv2d, generate_low_rank_linear


class Net(nn.Module):
    def __init__(self, ranks, scheme):
        super().__init__()
        self.conv1 = generate_low_rank_conv2d(3, 6, kernel_size=3, stride=1, padding=1, rank=ranks[0], scheme=scheme)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = generate_low_rank_conv2d(6, 12, kernel_size=3, stride=1, padding=1, rank=ranks[1], scheme=scheme)
        self.fc1 = generate_low_rank_linear(12 * 64 * 64, 120, ranks[0])
        self.fc2 = generate_low_rank_linear(120, 84, ranks[1])
        self.fc3 = generate_low_rank_linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x