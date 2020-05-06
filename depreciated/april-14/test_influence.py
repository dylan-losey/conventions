from models_influence import Convention
import pickle
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy


def plotter(model, s_star):
    s_star = torch.tensor([[s_star[0]],[s_star[1]]])
    xi_s, xi_z = model.rollout(s_star)
    time = np.linspace(0,1,model.n_steps)
    plt.plot(1,s_star[0,0], 'ks')
    plt.plot(time, xi_s[0,:])
    plt.plot(time[:-1], xi_z[0,:])


def main():

    modelname = "models/influence_a01.pt"
    model = Convention()
    model.alpha = 0.1
    model_dict = torch.load(modelname, map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval

    model.cost_Q()
    print(model.cost, model.diverg)

    model.traj_Q()

    # s_star = [0.5, 0.0]
    # plotter(model, s_star)
    # plt.show()


if __name__ == "__main__":
    main()
