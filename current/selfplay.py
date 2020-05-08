import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from clone import MLP
import pickle



class R_MLP(nn.Module):

    def __init__(self):
        super(R_MLP, self).__init__()

        self.human = MLP()
        model_dict = torch.load("models/h_model.pt", map_location='cpu')
        self.human.load_state_dict(model_dict)
        self.human.eval

        self.n_steps = 10
        self.fc_1 = nn.Linear(4, 8)
        self.fc_2 = nn.Linear(8, 8)
        self.fc_3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()

    def policy(self, x):
        h1 = torch.tanh(self.fc_1(x))
        h2 = torch.tanh(self.fc_2(h1))
        return self.fc_3(h2)

    def rollout(self, s_star, s_0):
        error = 0.0
        s = torch.FloatTensor(s_0)
        for t in range(self.n_steps):
            x = torch.cat((s, s_star), 0)
            ah = self.human.prediction(x).detach()
            context = torch.cat((s, ah), 0)
            ar = self.policy(context)
            s = s + 0.1 * ar
            error += 10.0 * torch.norm(ah - ar) + torch.norm(s - s_star)**2
        return error

    def loss(self):
        Q = 0.0
        g1 = torch.FloatTensor([1.0, 0.0])
        g2 = torch.FloatTensor([0.0, 1.0])
        for round in range(10):
            s_0 = np.random.random(2)
            for s_star in [g1, g2]:
                error = self.rollout(s_star, s_0)
                Q += error
        return Q


def main():

    EPOCH = 1000
    LR = 0.01
    LR_STEP_SIZE = 300
    LR_GAMMA = 0.1
    savename = "models/r_model.pt"

    model = R_MLP()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for idx in range(EPOCH):
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(idx, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
