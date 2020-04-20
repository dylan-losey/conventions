import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import copy
import numpy as np
import sys
import os
from inter_models import TEAM



class Model(object):

    def __init__(self, modelname):
        self.model = TEAM()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def traj(self, idx):
        xi_s, xi_ah, xi_ar = self.model.traj(idx)
        return xi_s, xi_ah, xi_ar


def plot_states(model):
    for idx in range(5):
        xi_s, xi_ah, xi_ar = model.traj(idx)
        plt.plot(xi_s)
    plt.show()


def plot_ah(model):
    for idx in range(5):
        xi_s, xi_ah, xi_ar = model.traj(idx)
        plt.plot(xi_ah)
    plt.show()


def plot_ar(model):
    for idx in range(5):
        xi_s, xi_ah, xi_ar = model.traj(idx)
        plt.plot(xi_ar)
    plt.show()


def main():

    modelname = 'models/team-1.pt'
    model = Model(modelname)

    plot_states(model)
    plot_ah(model)
    plot_ar(model)


if __name__ == "__main__":
    main()
