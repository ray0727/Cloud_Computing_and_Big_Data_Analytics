# HW1
code for HW1 `Music Memorability Prediction`


## Code Structure
    .
    ├── music-regression     # Sound Dataset
    ├── dataloader.py        # DataLoader
    ├── train.py             # training code
    ├── test.py              # evaluation
    └── README.md


## Training Model
```
$ python3 train.py
```
## Pretrained Model
`audio.pth`
## evaluation
* load audio.pth
```
$ python3 test.py --filename [audio file path]
```

