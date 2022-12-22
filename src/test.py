import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from src.models.CaffeBNAlexNet import CaffeBNAlexNet
from src.models.CaffeBNLowRankAlexNet import CaffeBNLowRankAlexNet
from src.ranks import ranks1,ranks2, ranks3,ranks4,ranks5,ranks6,ranks7,ranks8,ranks9
from src.visualization import imshow

def test(model_name, rank_=1, schema='scheme_1'):
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

    print(device)

    rank = []
    if rank_ == 1:
        rank = ranks1
    elif rank_ == 2:
        rank_ = ranks2
    elif rank_ == 3:
        rank_ = ranks3
    elif rank_ == 4:
        rank_ = ranks4
    elif rank_ == 5:
        rank_ = ranks5
    elif rank_ == 6:
        rank_ = ranks6
    elif rank_ == 7:
        rank_ = ranks7
    elif rank_ == 8:
        rank_ = ranks8
    elif rank_ == 9:
        rank_ = ranks9
    model = CaffeBNAlexNet()
    if model_name == 'CaffeBNAlexNet':
        model = CaffeBNAlexNet()
    else:
        if schema == 'scheme_2':
            model = CaffeBNLowRankAlexNet(ranks=rank, scheme='scheme_2')
        else:
            model = CaffeBNLowRankAlexNet(ranks=rank, scheme='scheme_1')
    model.load_state_dict(torch.load('.\trained_models'))

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

if __name__ == '__main__':
    train('CaffeBNAlexNet')
