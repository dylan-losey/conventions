import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle


class MotionData(Dataset):

    def __init__(self, filename):
        self.data = pickle.load(open(filename, "rb"))
        print("The dataset contains this many datapoints: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.FloatTensor(item)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.fc_1 = nn.Linear(8, 16)
        self.fc_2 = nn.Linear(16, 4)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h1 = torch.tanh(self.fc_1(x))
        return self.fc_2(h1)


def main():

    model = MLP()

    EPOCH = 1000
    BATCH_SIZE_RATIO = 10.0
    LR = 0.01
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.1

    dataname = "expert_dataset.pkl"
    savename = "expert_bc.pt"

    train_data = MotionData(dataname)
    BATCH_SIZE_TRAIN = int(len(train_data) / BATCH_SIZE_RATIO)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            s = x[:,0:8]
            a = x[:,8].long()
            ahat = model(s)
            loss = model.loss(ahat, a)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
