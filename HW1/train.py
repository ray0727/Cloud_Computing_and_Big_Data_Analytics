import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import SoundDataset
from torch.nn import init
import torchaudio
import pandas as pd
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=1)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


def train(model, train_loader, device):
    epochs = 100
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_loader)),
                                                epochs=epochs,
                                                anneal_strategy='linear')
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for i, (audio, score) in enumerate(train_loader):
            audio = audio.to(device)
            score = score.to(device)
            output = model(audio)
            score = score.view(-1,1)
            loss = criterion(output, score)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss = total_loss / len(train_loader)
        print(f'epoch{epoch:>2d} loss:{total_loss:.4f}')


if __name__== "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net()
    model = model.to(device)
    
    train_dataset = SoundDataset("./music-regression/audios/clips", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train(model=model, train_loader=train_loader, device=device)
    torch.save(model.state_dict(),"./audio.pth")