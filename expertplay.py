import gym
import torch
import numpy as np
from dqn import QNetwork
import sys
import pickle

def rollout(modelname):

    env = gym.make("LanderCustom-v0")
    qnetwork = QNetwork(state_size=8, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load(modelname))
    qnetwork.eval()

    episodes = 30
    scores = []

    for episode in range(episodes):

        force_x = +500.0
        env.start_state(force_x, 0.0)
        state = env.reset()
        score = 0

        while True:

            with torch.no_grad():
                state = torch.from_numpy(state).float()
                Q_values = qnetwork(state)
                action = np.argmax(Q_values.cpu().data.numpy())

            # env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                print("episode: ", episode, "score: ", score)
                break

        scores.append(score)

    env.close()
    print("The average score is: ", np.mean(np.array(scores)))
    return(scores)


def main():
    Q_threshold = 1000.0
    S = []
    for savenumber in range(10):
        modelname = "RL_" + str(Q_threshold) + "_" + str(savenumber) + ".pkl"
        scores = rollout(modelname)
        S.append(scores)
    print(S)




if __name__ == "__main__":
    main()
