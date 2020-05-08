import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import pickle


class MotionData(Dataset):

    def __init__(self, folder):
        self.data = []
        for filename in os.listdir(folder):
            local_data = np.array(pickle.load(open(folder + filename, "rb")))
            s_f = local_data[-1,0:2]
            g1 = local_data[-1,2:4]
            g2 = local_data[-1,4:6]
            s_star = list(g1)
            if np.linalg.norm(g1 - s_f) > np.linalg.norm(g2 - s_f):
                s_star = list(g2)
            for idx in range(len(local_data)):
                s = local_data[idx,0:2]
                a = local_data[idx,6:8]
                self.data.append(list(s)+s_star+list(a))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.FloatTensor(item)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.fc_1 = nn.Linear(4, 4)
        self.fc_2 = nn.Linear(4, 2)
        self.loss = nn.MSELoss()

    def prediction(self, x):
        h1 = self.fc_1(x)
        return self.fc_2(h1)


def main():

    model = MLP()

    EPOCH = 1000
    BATCH_SIZE_TRAIN = 50
    LR = 0.01
    LR_STEP_SIZE = 300
    LR_GAMMA = 0.1

    dataname = "baseline/"
    savename = "models/h_model.pt"

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            s = x[:,0:4]
            a = x[:,4:6]
            ahat = model.prediction(s)
            loss = model.loss(ahat, a)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
