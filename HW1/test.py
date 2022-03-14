import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import SoundDataset
from train import Net
import argparse
import torchaudio
import pandas as pd
import numpy as np

sub = {"track": [],
        "score": []}
df = pd.DataFrame(sub)



def evaluate(model, test_loader, device):
    count = 0
    with torch.set_grad_enabled(False):
        for batch_index, (audio, name) in enumerate(test_loader):
            audio = audio.to(device)
            predict = model(audio)
            for i in range(len(predict)):
                df.at[count, 'track'] = name[i]
                df.at[count, 'score'] = predict[i]
                count = count+1
            
                
            

if __name__== "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filename', default='./music-regression/audios/clips')
    args = parser.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = SoundDataset(args.filename, mode="test")
    
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)
    model = Net()
    model.load_state_dict(torch.load("./audio.pth"))
    model=model.to(device)
    evaluate(model, test_loader, device)
    # print(df)
    df.to_csv("submission.csv", index=False)