# HW3
code for HW3 `Anomaly Detection`

Method: AutoEncoder


## Code Structure
    .
    ├── csv files            # Dataset (normal, public, private)
    ├── dataloader.py        # Load Dataset
    ├── model.py             # Model Structure
    ├── train.py             # training code
    ├── test.py              # evaluation and generate submission file
    └── README.md


## Training Model
Show the training loss, and the roc score of the public dataset
change csv file name in the code based on which sensor you are going to train
```
python3 train.py
```
## Pretrained Model
`sensor_A.pth, sensor_B.pth, sensor_C.pth, sensor_D.pth, sensor_E.pth`

## evaluation
* load pretrained model and generate submission csv file
```
python3 test.py 
```