import gym
import torch
import numpy as np
from dqn import QNetwork
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import copy
import os


class MotionData(Dataset):

    def __init__(self, dataset):
        self.data = dataset
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


def train(dataset, modelname, savename):

    model = MLP()
    LR = 0.01
    EPOCH = 400
    if modelname is not None:
        model.load_state_dict(torch.load(modelname))

    BATCH_SIZE_TRAIN = 500

    train_data = MotionData(dataset)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            s = x[:,0:8]
            a = x[:,8].long()
            ahat = model(s)
            loss = model.loss(ahat, a)
            loss.backward()
            optimizer.step()
        # print(epoch, loss.item())
    torch.save(model.state_dict(), savename)


def store(modelname, savename):
    model = MLP()
    model.load_state_dict(torch.load(modelname))
    torch.save(model.state_dict(), savename)


def rollout(Q_threshold, modelname):

    if modelname is not None:
        human = MLP()
        human.load_state_dict(torch.load(modelname))
        human.eval()

    state = env.reset()
    score = 0
    dataset = []

    while True:

        # get robot and human actions
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            Q_values = qnetwork(state).data.numpy()
            action_star = np.argmax(Q_values)
            action = np.random.choice(np.arange(4))
            if modelname is not None:
                action_pred_dist = softmax(human(state).data).numpy()
                action = np.random.choice(np.arange(4), p=action_pred_dist)

        # save data
        loss = Q_values[action_star] - Q_values[action]
        dataset.append(list(state.numpy()) + [action, loss, action_star])

        # shared autonomy
        if loss > Q_threshold:
            action = action_star

        # update environment
        # env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break

    env.close()
    return score, dataset


def target(modelname, n_episodes):
    score = 0.0
    for idx in range(n_episodes):
        s, _ = rollout(1e3, modelname)
        score += s / float(n_episodes)
    return score


def balance_human(Q_threshold, modelname, n_episodes):
    dataset = []
    for idx in range(n_episodes):
        score, data = rollout(Q_threshold, modelname)
        for item in data:
            state = item[0:8]
            action = item[8]
            loss = item[9]
            action_star = item[10]
            dataset.append(state + [action_star])
    return dataset


def success_human(Q_threshold, modelname, n_episodes):
    dataset = []
    for idx in range(n_episodes):
        score, data = rollout(Q_threshold, modelname)
        if score > 100:
            for item in data:
                state = item[0:8]
                action = item[8]
                loss = item[9]
                action_star = item[10]
                dataset.append(state + [action_star])
    return dataset


def fail_human(Q_threshold, modelname, n_episodes):
    dataset = []
    for idx in range(n_episodes):
        score, data = rollout(Q_threshold, modelname)
        if score < 100:
            for item in data:
                state = item[0:8]
                action = item[8]
                loss = item[9]
                action_star = item[10]
                dataset.append(state + [action_star])
    return dataset



def main():

    humanmodel = None
    humandata = []
    humanscore = []
    tempname = 'test.pt'

    n_episodes = 5

    mdp0 = [0.0]
    mdp1 = [1.0]
    mdp2 = [5.0]
    mdp3 = [10.0]
    mdp4 = [20.0]
    mdp5 = [40.0]
    mdp6 = [1e3]

    curriculum = [mdp3, mdp3, mdp3, mdp3, mdp3]

    for count, M in enumerate(curriculum):

        dataset_M = balance_human(M[0], humanmodel, n_episodes)
        n_datapoints = int(len(dataset_M) / 50.0)
        dataset_M = random.sample(dataset_M, k=min(n_datapoints,len(dataset_M)))
        humandata_M = humandata + dataset_M

        train(humandata_M, humanmodel, tempname)
        score_M = target(tempname, n_episodes)
        store(tempname, 'eval.pt')
        print("MDP: ", M, "Score: ", score_M)

        humanmodel = 'eval.pt'
        humanscore.append(score_M)
        humandata = copy.deepcopy(humandata_M)

    print("Full scores: ", humanscore)
    return humanscore


if __name__ == "__main__":

    env = gym.make("LunarLander-v2")
    qnetwork = QNetwork(state_size=8, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load('dqn_expert.pth'))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=0)

    SCORES = []
    n_runs = int(sys.argv[1])
    for idx in range(n_runs):
        print("ROUND #: ", idx)
        humanscore = main()
        SCORES.append(humanscore)
    print("Overall: ", SCORES)
