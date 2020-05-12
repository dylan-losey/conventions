import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle



class MLP_MLP(nn.Module):

    def __init__(self):
        super(MLP_MLP, self).__init__()

        self.name = "models/sp_robot.pt"

        self.fc_1 = nn.Linear(4, 4)
        self.fc_2 = nn.Linear(4, 2)

        self.rc_1 = nn.Linear(4, 4)
        self.rc_2 = nn.Linear(4, 2)

    def human(self, x):
        h1 = torch.tanh(self.fc_1(x))
        return self.fc_2(h1)

    def robot(self, x):
        h1 = torch.tanh(self.rc_1(x))
        return self.rc_2(h1)


    def loss(self, n_samples):
        loss = 0.0
        for iteration in range(n_samples):
            state = torch.rand(2)
            target = torch.rand(2)
            error = target - state
            ah_star = torch.FloatTensor([-error[1], error[0]])
            ah = self.human(torch.cat((state, target), 0))
            ar = self.robot(torch.cat((state, ah), 0))
            state += 1.0 * ar
            loss += torch.norm(target - state)**2
            loss += torch.norm(ah_star - ah)**2
            loss += abs(torch.norm(ah) - torch.norm(ar))
        return loss



def main():

    EPOCH = 1000
    LR = 0.01
    LR_STEP_SIZE = 400
    LR_GAMMA = 0.1
    savename = "models/sp_robot.pt"

    model = MLP_MLP()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for idx in range(EPOCH):
        optimizer.zero_grad()
        loss = model.loss(100)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(idx, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
