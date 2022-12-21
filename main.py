import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from collections import OrderedDict
from models import CaffeBNAlexNet, CaffeBNLowRankAlexNet, CNN, CNNLowRank
import time
import datetime

transform = transforms.Compose(
                                [transforms.Resize((256,256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CaffeBNAlexNet().to(device)

#For CNNLowRank
#ranks = [2, 12]
#model = CNNLowRank(ranks, scheme='scheme_1')

#For CaffeBNLowRankAlexNet
#Scheme 2
#ranks1 = [7, 28, 69, 71, 88, 506, 349, 255]
#ranks2 = [8, 28, 76, 79, 99, 606, 428, 288]
#ranks3 = [8, 30, 85, 90, 114, 770, 582, 341]
#ranks4 = [9, 34, 103, 116, 143, 1075, 937, 449]
#ranks5 = [11, 46, 148, 172, 199, 1790, 1733, 694]
#
##Scheme 1
#ranks6 = [25, 31, 76, 62, 76, 602, 429, 290]
#ranks7 = [26, 34, 79, 66, 82, 688, 507, 318]
#ranks8 = [27, 33, 82, 71, 87, 760, 586, 344]
#ranks9 = [33, 53, 133, 134, 156, 1764, 1748, 703]
#
#model = CaffeBNLowRankAlexNet(ranks=ranks1, scheme='scheme_1')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9) 

start_time = time.time()
for epoch in range(2): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
elapsed_time = time.time() - start_time
print(str(datetime.timedelta(seconds=elapsed_time)))

