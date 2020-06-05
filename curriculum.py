import gym
import torch
import numpy as np
from dqn import QNetwork
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle
import random
import copy


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



def rollout(force_x, Q_threshold, modelname):

    env = gym.make("LanderCustom-v0")
    qnetwork = QNetwork(state_size=8, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load('basic_lander.pth'))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=0)

    if modelname is not None:
        human = MLP()
        human.load_state_dict(torch.load(modelname))
        human.eval()

    env.start_state(force_x, 0.0)
    state = env.reset()
    score = 0
    dataset = []

    while True:

        with torch.no_grad():
            state = torch.from_numpy(state).float()
            Q_values = qnetwork(state).data.numpy()
            action_star = np.argmax(Q_values)
            if modelname is not None:
                action_pred_dist = softmax(human(state).data).numpy()
                action = np.random.choice(np.arange(4), p=action_pred_dist)
            else:
                action = np.random.choice(np.arange(4))

        loss = Q_values[action_star] - Q_values[action]
        dataset.append(list(state.numpy()) + [action, loss, action_star])

        # shared autonomy
        if loss > Q_threshold:
            action = action_star

        # env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            # print("score: ", score)
            break

    env.close()
    return score, dataset


def target(modelname):
    score = 0.0
    n_episodes = 10
    for idx in range(n_episodes):
        score1, _ = rollout(0.0, 1e3, modelname)
        score2, _ = rollout(+500.0, 1e3, modelname)
        score3, _ = rollout(-500.0, 1e3, modelname)
        score += score1 + score2 + score3
    score /= (3.0 * n_episodes)
    return score


def success_human(force_x, Q_threshold, modelname):
    dataset = []
    n_episodes = 5
    for idx in range(n_episodes):
        score, data = rollout(force_x, Q_threshold, modelname)
        if score > 100:
            for item in data:
                state = item[0:8]
                action = item[8]
                loss = item[9]
                action_star = item[10]
                dataset.append(state + [action_star])
    return dataset


def fail_human(force_x, Q_threshold, modelname):
    dataset = []
    n_episodes = 5
    for idx in range(n_episodes):
        score, data = rollout(force_x, Q_threshold, modelname)
        if score < 100:
            for item in data:
                state = item[0:8]
                action = item[8]
                loss = item[9]
                action_star = item[10]
                dataset.append(state + [action_star])
    return dataset


def train(dataset, modelname, savename):

    model = MLP()
    LR = 0.01
    EPOCH = 100
    if modelname is not None:
        LR = 0.01
        EPOCH = 100
        model.load_state_dict(torch.load(modelname))

    BATCH_SIZE_TRAIN = 1000

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
    torch.save(model.state_dict(), savename)


def store(modelname, savename):
    model = MLP()
    model.load_state_dict(torch.load(modelname))
    torch.save(model.state_dict(), savename)


def main():

    humanmodel = None
    humandata = []
    humanscore = 0


    mdp1 = [0.0, 1.0]
    mdp2 = [0.0, 7.5]
    mdp3 = [0.0, 20.0]

    while humanscore < 200:

        dataset1 = fail_human(mdp1[0], mdp1[1], humanmodel)
        dataset2 = fail_human(mdp2[0], mdp2[1], humanmodel)
        dataset3 = fail_human(mdp3[0], mdp3[1], humanmodel)

        dataset1 = random.sample(dataset1, k=min(500,len(dataset1)))
        dataset2 = random.sample(dataset2, k=min(500,len(dataset2)))
        dataset3 = random.sample(dataset3, k=min(500,len(dataset3)))

        humandata1 = humandata + dataset1
        humandata2 = humandata + dataset2
        humandata3 = humandata + dataset3

        if len(humandata1) > 0:
            train(humandata1, humanmodel, 'test1.pt')
            score1 = target('test1.pt')
        else:
            score1 = -np.Inf

        if len(humandata2) > 0:
            train(humandata2, humanmodel, 'test2.pt')
            score2 = target('test2.pt')
        else:
            score2 = -np.Inf

        if len(humandata3) > 0:
            train(humandata3, humanmodel, 'test3.pt')
            score3 = target('test3.pt')
        else:
            score3 = -np.Inf

        if score1 > score2 and score1 > score3:
            store('test1.pt', 'eval.pt')
            humandata = copy.deepcopy(humandata1)
            humanscore = score1
            print('#1 is best')

        elif score2 > score1 and score2 > score3:
            store('test2.pt', 'eval.pt')
            humandata = copy.deepcopy(humandata2)
            humanscore = score2
            print('#2 is best')

        elif score3 > score1 and score3 > score2:
            store('test3.pt', 'eval.pt')
            humandata = copy.deepcopy(humandata3)
            humanscore = score3
            print('#3 is best')

        humanmodel = 'eval.pt'

        print('mdp1: ', score1)
        print('mpd2: ', score2)
        print('mpd3: ', score3)



if __name__ == "__main__":
    main()
