import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import Convention
import pickle
import sys


savename = "MLP-FIX.pt"
model = Convention()

LR = 0.1
LR_STEP_SIZE = 300
LR_GAMMA = 0.1

def main():

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)


    for count in range(900):
        model.zero_grad()
        model.init_hidden()
        loss = model.cost_Q()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(count, loss.item())

    torch.save(model.state_dict(), "models/" + savename)
    model.eval
    model.traj_Q()


if __name__ == "__main__":
    main()
