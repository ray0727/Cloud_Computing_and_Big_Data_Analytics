import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from model import SimCLR
from dataloader import ImageDataset, evaluatedata, ImageNPY
import matplotlib.pyplot as plt
import time

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simclr_model = SimCLR(num_class=4).to(DEVICE)
simclr_model.load_state_dict(torch.load("model.pth"))

simclr_model.eval()   
simclr_model.linear_eval = True 
npy_dataset = ImageNPY()
numpy_embedding = np.empty((0,512), dtype=np.float32)
count = 0
for image in npy_dataset:
    with torch.no_grad():
        count = count+1
        print(count)
        image = image.unsqueeze(0)
        image = image.to(DEVICE)
        embedding, _ = simclr_model(image)
        embedding = embedding.cpu().numpy()
        # print("one", embedding.shape)
        numpy_embedding = np.append(numpy_embedding, embedding, axis=0)
        # print("append", numpy_embedding.shape)

print("save numpy", numpy_embedding.shape)
np.save("./309512074.npy", numpy_embedding)

test_dataset = evaluatedata()
test_dataset = test_dataset.get_data()
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