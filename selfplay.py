import gym
import time
import torch
import numpy as np
from dqn import QNetwork
import sys
from clone import MLP


def main():

    env = gym.make("LanderCustom-v0")

    qnetwork = QNetwork(state_size=8, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load('basic_lander.pth'))
    qnetwork.eval()

    human = MLP()
    human.load_state_dict(torch.load('mlp_model.pt'))
    human.eval()

    fx_init = float(sys.argv[1])
    Q_threshold = float(sys.argv[2])

    episodes = 100
    scores = []
    env.start_state(fx_init, 0)

    for episode in range(episodes):

        state = env.reset()
        score = 0

        while True:

            with torch.no_grad():
                state = torch.from_numpy(state).float()
                Q_values = qnetwork(state).data.numpy()
                action_pred_dist = human(state).data.numpy()
            action_star = np.argmax(Q_values)
            action = np.argmax(action_pred_dist)

            # action = np.random.randint(0, 4)

            loss = Q_values[action_star] - Q_values[action]
            if loss > Q_threshold:
                action = action_star

            # env.render()
            state, reward, done, _ = env.step(action)
            score += reward

            if done:
                print(episode, score)
                break

        scores.append(score)

    env.close()
    # print(scores)
    print(np.mean(np.array(scores)))


if __name__ == "__main__":
    main()
