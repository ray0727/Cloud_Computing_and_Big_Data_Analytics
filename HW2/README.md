# HW2
code for HW2 `Self Supervised Learning`

Method: SimCLR


## Code Structure
    .
    ├── hw2                  # Image Dataset (Unlabeled, testing)
    ├── dataloader.py        # DataLoader
    ├── model.py             # Model Structure
    ├── view_generator.py    # image augmentation
    ├── train.py             # training code
    ├── test.py              # evaluation
    └── README.md


## Training Model
Show the training loss, and the KNN accuracy of the testing dataset
```
$ python3 train.py
```
## Pretrained Model
`model.pth`
## Saved Embeddings
`309512074.npy`
## evaluation
* load model.pth and 
save embeddings to 309512074.npy
```
$ python3 test.py 
```