import gym
import torch
import numpy as np
from dqn import QNetwork



def main():

    env = gym.make("LanderCustom-v0")

    qnetwork = QNetwork(state_size=8, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load('basic_lander.pth'))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=0)

    episodes = 20
    scores = []
    env.start_state(0, 1000)

    for episode in range(episodes):

        state = env.reset()
        score = 0

        while True:

            with torch.no_grad():
                state = torch.from_numpy(state).float()
                Q_values = qnetwork(state)
                action = np.argmax(Q_values.cpu().data.numpy())

            env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break

        scores.append(score)

    env.close()
    print(scores)


if __name__ == "__main__":
    main()
