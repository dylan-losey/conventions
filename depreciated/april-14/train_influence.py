import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models_influence import Convention
import matplotlib.pyplot as plt
import pickle
import sys


savename = "influence.pt"
model = Convention()
model.alpha = 10

LR = 0.1
LR_STEP_SIZE = 300
LR_GAMMA = 0.5

def main():

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)


    for count in range(900):
        model.zero_grad()
        loss = model.cost_Q()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(count, loss.item())

    torch.save(model.state_dict(), "models/" + savename)
    model.eval

    plt.plot(model.cost)
    print(model.cost[-1])
    plt.show()

    print(model.diverg[-1])
    plt.plot(model.diverg)
    plt.show()

    model.traj_Q()


if __name__ == "__main__":
    main()
