import gym
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
    human.load_state_dict(torch.load('expert_bc.pt'))
    human.eval()
    softmax = torch.nn.Softmax(dim=0)

    episodes = 30
    scores = []
    Q_threshold = 1e-2

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
                Q_values = qnetwork(state).data.numpy()
                action_pred_dist = softmax(human(state).data).numpy()
            action_star = np.argmax(Q_values)
            action = np.random.choice(np.arange(4), p=action_pred_dist)

            loss = Q_values[action_star] - Q_values[action]
            # if loss > Q_threshold:
            #     action = action_star

            # env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                print("episode: ", episode, "score: ", score)
                break

        scores.append(score)

    env.close()
    print("The average score is: ", np.mean(np.array(scores)))


if __name__ == "__main__":
    main()
