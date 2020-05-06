import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from cloning import MLP, MotionData


class Model(object):

    def __init__(self, modelname):
        self.model = MLP()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval


def correct_predictions(model, data):
    n_correct = 0
    for item in data:
        s = item[0:2]
        a = item[2].long()
        ahat = model.model.prediction(s)
        ahat = F.softmax(ahat.detach(), dim=0)
        max, i_max = ahat.max(0)
        if a == i_max:
            n_correct += 1
    print("total correct predictions: ", n_correct)
    print("percent correct predictions: ", round(n_correct / len(data) * 100))


def compare_models(model1, model2, data):
    pred1 = np.zeros((len(data), 1))
    pred2 = np.zeros((len(data), 1))
    for count, item in enumerate(data):
        s = item[0:2]
        a = item[2].long()
        ahat1 = model1.model.prediction(s)
        ahat2 = model2.model.prediction(s)
        ahat1 = F.softmax(ahat1.detach(), dim=0)
        ahat2 = F.softmax(ahat2.detach(), dim=0)
        _, i_max_1 = ahat1.max(0)
        _, i_max_2 = ahat2.max(0)
        pred1[count, :] = i_max_1
        pred2[count, :] = i_max_2
    diff_pred = 0
    for count in range(len(pred1)):
        if abs(pred1[count] - pred2[count]) > 0.5:
            diff_pred += 1
    print("total different predictions: ", diff_pred)
    print("percent different predictions: ", round(diff_pred / len(data) * 100))



def main():

    modelname1 = "mlp_model_1.pt"
    model1 = Model(modelname1)
    modelname2 = "mlp_model_2.pt"
    model2 = Model(modelname2)

    dataname = "duckGame2_reduced.npy"
    data = MotionData(dataname)

    correct_predictions(model1, data)
    correct_predictions(model2, data)
    compare_models(model1, model2, data)



if __name__ == "__main__":
    main()
