import gym
import torch
import numpy as np
from dqn import QNetwork
import sys
import pickle


def main():

    env = gym.make("LanderCustom-v0")
    savename = 'expert_dataset.pkl'

    qnetwork = QNetwork(state_size=8, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load('basic_lander.pth'))
    qnetwork.eval()

    episodes = 30
    scores = []
    dataset = []

    for episode in range(episodes):

        if episode < 10:
            force_x = 0.0
        elif episode < 20:
            force_x = +500.0
        else:
            force_x = -500.0

        env.start_state(force_x, 0.0)
        state = env.reset()
        score = 0

        while True:

            with torch.no_grad():
                state = torch.from_numpy(state).float()
                Q_values = qnetwork(state)
                action = np.argmax(Q_values.cpu().data.numpy())

            dataset.append(list(state) + [action])

            # env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                print("episode: ", episode, "score: ", score)
                break

        scores.append(score)

    env.close()
    pickle.dump(dataset, open(savename, "wb"))
    print("The average score is: ", np.mean(np.array(scores)))


if __name__ == "__main__":
    main()
