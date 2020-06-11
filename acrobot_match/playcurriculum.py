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
import os


def rollout(Q_threshold, data):
    humandata = np.asarray(data)
    state = env.reset()
    experiences = []
    score = 0
    for timestep in range(500):
        with torch.no_grad():
            state_t = torch.from_numpy(state).float()
            Q_values = qnetwork(state_t).data.numpy()
            action_star = np.argmax(Q_values)
        action = np.random.choice(np.arange(3))
        if len(humandata) > 0:
            idx = np.linalg.norm(humandata[:,0:6] - state, axis=1).argmin()
            action = int(humandata[idx,6])
        loss = Q_values[action_star] - Q_values[action]
        experiences.append(list(state) + [loss, 1-(loss>Q_threshold), action, action_star])
        if loss > Q_threshold:
            action = action_star
        # env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
    env.close()
    return score, experiences


def experience_matching():
    mydata = []
    mytrainscore = []
    mytestscore = []
    n_episodes = 15
    sa = 0.0#0.5
    for count in range(n_episodes):
        score_train, experiences = rollout(sa, mydata)
        score_test, _ = rollout(1e3, mydata)
        mytrainscore.append(score_train)
        mytestscore.append(score_test)
        learned_experiences = []
        corrections = 0
        for item in experiences:
            state, loss, accepted, action, action_star = item[0:6], item[6], item[7], item[8], item[9]
            if not accepted:
                corrections += 1
            if accepted or not accepted:
                learned_experiences.append(state + [action_star])
        if len(learned_experiences) > 20:
            learned_experiences = random.sample(learned_experiences, k=20)
        mydata = mydata + learned_experiences
        print(count, " TrainScore: ", score_train, " TestScore: ", \
            score_test, " Correction: ", corrections, " Data: ", len(mydata))
    return mytrainscore, mytestscore


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    env.seed(0)
    qnetwork = QNetwork(state_size=6, action_size=3, seed=0)
    qnetwork.load_state_dict(torch.load('models/dqn_acrobot.pth'))
    qnetwork.eval()
    scores = []
    for idx in range(25):
        mytrainscore, mytestscore = experience_matching()
        scores.append(mytestscore)
    pickle.dump( scores, open( "results/obs/AllSA-Full.pkl", "wb" ) )
