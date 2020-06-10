import gym
import torch
import numpy as np
from dqn import QNetwork
import sys
import pickle


def rollout(modelname, n_episodes):

    qnetwork.load_state_dict(torch.load(modelname))
    qnetwork.eval()
    scores = []
    max_t = 200

    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        for t in range(max_t):
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                Q_values = qnetwork(state)
                action = np.argmax(Q_values.cpu().data.numpy())
            # env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)

    env.close()
    avg_score = np.mean(np.asarray(scores))
    std_score = np.std(np.asarray(scores))
    print(modelname + ":", str(avg_score) + " +- " + str(std_score))
    return(scores)


if __name__ == "__main__":
    modeltype = sys.argv[1]
    savename = "results/" + modeltype + ".pkl"
    env = gym.make("CartPole-v0")
    qnetwork = QNetwork(state_size=4, action_size=2, seed=100)
    S = []
    n_episodes = 20
    for savenumber in range(15):
        modelname = "models/" + modeltype + "_" + str(savenumber) + ".pth"
        scores = rollout(modelname, n_episodes)
        S.append(scores)
    pickle.dump( S, open( savename, "wb" ) )
