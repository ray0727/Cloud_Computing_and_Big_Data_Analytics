import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from model import SimCLR
from dataloader import ImageDataset, evaluatedata, ImageNPY
import matplotlib.pyplot as plt
import time

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')

def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            # print(knn)
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)

LABELS = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
LABELS = LABELS.to(DEVICE) #128,128

def cont_loss(features, temp):
    """
    The NTxent Loss.
    
    Args:
        z1: The projection of the first branch
        z2: The projeciton of the second branch
    
    Returns:
        the NTxent loss
    """
    similarity_matrix = torch.matmul(features, features.T) # 128, 128
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(DEVICE)
    # ~mask is the negative of the mask
    # the view is required to bring the matrix back to shape
    labels = LABELS[~mask].view(LABELS.shape[0], -1) # 128, 127
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 128, 127

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # 128, 1

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # 128, 126

    logits = torch.cat([positives, negatives], dim=1) # 128, 127
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / temp
    return logits, labels


train_dataset = ImageDataset("./hw2/unlabeled", n_views=2)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
simclr_model = SimCLR(num_class=4).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(simclr_model.parameters(), lr=0.00003, weight_decay=0.01)
EPOCHS = 30

test_dataset = evaluatedata()
test_dataset = test_dataset.get_data()
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=500, shuffle=True, drop_last=True, pin_memory=True)

for epoch in range(EPOCHS):
    simclr_model.train()
    simclr_model.linear_eval = False
    t0 = time.time()
    running_loss = 0.0
    for i, views in enumerate(train_loader):
        embedding, projections = simclr_model([view.to(DEVICE) for view in views])
        # print(embedding.shape)
        logits, labels = cont_loss(projections, temp=2)
        # print("label", labels)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print(f'EPOCH: {epoch+1} BATCH: {i+1} LOSS: {(running_loss/100):.4f} ')
            # acc = KNN(emb=embedding, cls=labels, batch_size=BATCH_SIZE)
            # print("Accuracy: %.5f" % acc)
            running_loss = 0.0
    print(f'Time taken: {((time.time()-t0)/60):.3f} mins')
    correct=0
    simclr_model.linear_eval = True
    simclr_model.eval()
    images = torch.tensor([])
    labels = torch.tensor([])
    for image, label in test_dataset:
        image = image.unsqueeze(0)
        images = torch.cat((images,image), dim=0)
        label = torch.tensor(np.array([label]))
        labels = torch.cat((labels, label), dim=0)
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    test_embedding, pred = simclr_model(images)
    acc = KNN(emb=test_embedding, cls=labels, batch_size=500)
    print("Testing KNN: %.5f" % acc)

torch.save(simclr_model.state_dict(), "./model.pth")
