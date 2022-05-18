import torch
import torch.nn as nn
import torchvision
from dataloader import create_dataset
from model import RecurrentAutoencoder
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import roc_auc_score


def train(model, train_dataset, val_dataset, val_label, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0
    for epoch in range(1, n_epochs+1):
        model = model.train()
        train_losses = []
        for seq in train_dataset:
            optimizer.zero_grad()
            seq = seq.to(device)
            seq_pred = model(seq)
            loss = criterion(seq_pred, seq)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model = model.eval()
        with torch.no_grad():
            for val_seq in val_dataset:
                val_seq = val_seq.to(device)
                val_seq_pred = model(val_seq)
                reconstruction_error = abs(val_seq_pred - val_seq)
                score = roc_auc_score(val_label, reconstruction_error.cpu().detach().numpy())
        # if score > best_score and epoch >=20:
        #     best_score = score
        #     best_model_wts = copy.deepcopy(model.state_dict())
        train_loss = np.mean(train_losses)
        print(f'Epoch {epoch}: train loss {train_loss} roc score {score}')
        # print(f'Epoch {epoch}: train loss {train_loss} reconstruction error {reconstruction_error} roc score {score}')
    # model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "./sensor_E.pth")

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_pd = pd.read_csv('./sensor_E_normal.csv')
    training_value = training_pd.telemetry
    training_value = training_value.to_numpy()
    for i in range(4000-len(training_value)):
        training_value = np.append(training_value, training_value[i])
    public_pd = pd.read_csv('./sensor_E_public.csv')
    public_value = public_pd.telemetry
    public_value = public_value.to_numpy()
    public_label = public_pd.label
    public_label = public_label.to_numpy()
    train_dataset, seq_len, n_features = create_dataset(training_value)
    val_dataset, _, _ = create_dataset(public_value)
    model = RecurrentAutoencoder(seq_len, n_features, 32)
    model = model.to(device)
    train(model, train_dataset, val_dataset, public_label, 75)