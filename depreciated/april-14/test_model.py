from models import Convention
import pickle
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy


def main():

    modelname = "models/" + sys.argv[1]
    model = Convention()
    model_dict = torch.load(modelname, map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval

    print(model.cost_Q())
    model.traj_Q()


if __name__ == "__main__":
    main()
