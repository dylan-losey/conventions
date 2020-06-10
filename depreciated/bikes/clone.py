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
            local_data = pickle.load(open(folder + filename, "rb"))
            for item in local_data:
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.FloatTensor(item)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.name = "models/bc_human.pt"
        self.fc_1 = nn.Linear(4, 4)
        self.fc_2 = nn.Linear(4, 2)
        self.loss = nn.MSELoss()

    def human(self, x):
        h1 = torch.tanh(self.fc_1(x))
        return self.fc_2(h1)


def main():

    model = MLP()

    EPOCH = 1000
    BATCH_SIZE_TRAIN = 100
    LR = 0.01
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.1

    dataname = "baseline/"
    savename = "models/bc_human.pt"

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            s = x[:,0:4]
            a = x[:,4:6]
            ahat = model.human(s)
            loss = model.loss(ahat, a)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item(), a[0,:], ahat[0,:].detach())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
