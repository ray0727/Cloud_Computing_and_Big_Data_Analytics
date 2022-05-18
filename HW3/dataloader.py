import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd



def create_dataset(data):
    # print(data)
    dataset = [torch.tensor(data).unsqueeze(1).float()]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features

# training_pd = pd.read_csv('./sensor_A_normal.csv')
# training_value = training_pd.telemetry
# training_value = training_value.to_numpy()
# plt.plot(training_value, label = 'Data')
# plt.show()

# for i in range(4000-len(training_value)):
#     training_value = np.append(training_value, training_value[i])

# plt.plot(training_value, label = 'Data')
# plt.show()