import torch
import torch.nn as nn
import torchvision
from dataloader import create_dataset
from model import RecurrentAutoencoder
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import roc_auc_score

sub = {"id": [], "pred": []}
df = pd.DataFrame(sub)

count = 0
def evaluate_public(model, val_dataset, val_label, device):
    global count
    with torch.no_grad():
        for val_seq in val_dataset:
            val_seq = val_seq.to(device)
            val_seq_pred = model(val_seq)
            reconstruction_error = abs(val_seq_pred - val_seq)
            for i in range(len(reconstruction_error)):
                df.at[count, 'id'] = count
                df.at[count, 'pred'] = reconstruction_error[i]
                count = count+1

            score = roc_auc_score(val_label, reconstruction_error.cpu().detach().numpy())
            print("roc score: ", score)
        
def evaluate_private(model, val_dataset, device):
    global count
    with torch.no_grad():
        for val_seq in val_dataset:
            val_seq = val_seq.to(device)
            val_seq_pred = model(val_seq)
            reconstruction_error = abs(val_seq_pred - val_seq)
            for i in range(len(reconstruction_error)):
                df.at[count, 'id'] = count
                df.at[count, 'pred'] = reconstruction_error[i]
                count = count+1

            

if __name__== "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    public_pd = pd.read_csv('./sensor_A_public.csv')
    public_value = public_pd.telemetry
    public_value = public_value.to_numpy()
    public_label = public_pd.label
    public_label = public_label.to_numpy()
    public_dataset, seq_len, n_features = create_dataset(public_value)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model.load_state_dict(torch.load("./sensor_A.pth"))
    model=model.to(device)
    model = model.eval()

    evaluate_public(model, public_dataset, public_label, device)

    public_pd = pd.read_csv('./sensor_B_public.csv')
    public_value = public_pd.telemetry
    public_value = public_value.to_numpy()
    public_label = public_pd.label
    public_label = public_label.to_numpy()
    public_dataset, seq_len, n_features = create_dataset(public_value)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model.load_state_dict(torch.load("./sensor_B.pth"))
    model=model.to(device)
    model = model.eval()

    evaluate_public(model, public_dataset, public_label, device)

    public_pd = pd.read_csv('./sensor_C_public.csv')
    public_value = public_pd.telemetry
    public_value = public_value.to_numpy()
    public_label = public_pd.label
    public_label = public_label.to_numpy()
    public_dataset, seq_len, n_features = create_dataset(public_value)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model.load_state_dict(torch.load("./sensor_C.pth"))
    model=model.to(device)
    model = model.eval()

    evaluate_public(model, public_dataset, public_label, device)

    public_pd = pd.read_csv('./sensor_D_public.csv')
    public_value = public_pd.telemetry
    public_value = public_value.to_numpy()
    public_label = public_pd.label
    public_label = public_label.to_numpy()
    public_dataset, seq_len, n_features = create_dataset(public_value)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model.load_state_dict(torch.load("./sensor_D.pth"))
    model=model.to(device)
    model = model.eval()

    evaluate_public(model, public_dataset, public_label, device)

    public_pd = pd.read_csv('./sensor_E_public.csv')
    public_value = public_pd.telemetry
    public_value = public_value.to_numpy()
    public_label = public_pd.label
    public_label = public_label.to_numpy()
    public_dataset, seq_len, n_features = create_dataset(public_value)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model.load_state_dict(torch.load("./sensor_E.pth"))
    model=model.to(device)
    model = model.eval()

    evaluate_public(model, public_dataset, public_label, device)

    private_pd = pd.read_csv('./sensor_A_private.csv')
    private_value = private_pd.telemetry
    private_value = private_value.to_numpy()

    private_dataset, _, _ = create_dataset(private_value)
    model.load_state_dict(torch.load("./sensor_A.pth"))
    model=model.to(device)
    model = model.eval()
    evaluate_private(model, private_dataset, device)

    private_pd = pd.read_csv('./sensor_B_private.csv')
    private_value = private_pd.telemetry
    private_value = private_value.to_numpy()

    private_dataset, _, _ = create_dataset(private_value)
    model.load_state_dict(torch.load("./sensor_B.pth"))
    model=model.to(device)
    model = model.eval()
    evaluate_private(model, private_dataset, device)

    private_pd = pd.read_csv('./sensor_C_private.csv')
    private_value = private_pd.telemetry
    private_value = private_value.to_numpy()

    private_dataset, _, _ = create_dataset(private_value)
    model.load_state_dict(torch.load("./sensor_C.pth"))
    model=model.to(device)
    model = model.eval()
    evaluate_private(model, private_dataset, device)

    private_pd = pd.read_csv('./sensor_D_private.csv')
    private_value = private_pd.telemetry
    private_value = private_value.to_numpy()

    private_dataset, _, _ = create_dataset(private_value)
    model.load_state_dict(torch.load("./sensor_D.pth"))
    model=model.to(device)
    model = model.eval()
    evaluate_private(model, private_dataset, device)

    private_pd = pd.read_csv('./sensor_E_private.csv')
    private_value = private_pd.telemetry
    private_value = private_value.to_numpy()

    private_dataset, _, _ = create_dataset(private_value)
    model.load_state_dict(torch.load("./sensor_E.pth"))
    model=model.to(device)
    model = model.eval()
    evaluate_private(model, private_dataset, device)

    df['id'] = df['id'].apply(np.int64)
    df.to_csv("./result/submission.csv", index=False)


    
