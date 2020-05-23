import gym
import pygame
import time
import torch
import sys
import pickle
import numpy as np
from dqn import QNetwork


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        z2 = -self.gamepad.get_axis(1)
        action = 0
        if abs(z1) > abs(z2):
            if z1 > self.DEADBAND:
                action = 3
            if z1 < -self.DEADBAND:
                action = 1
        else:
            if z2 > self.DEADBAND:
                action = 2
        start = self.gamepad.get_button(0)
        stop = self.gamepad.get_button(7)
        return action, start, stop


def main():

    env = gym.make("LanderCustom-v0")
    fx_init = float(sys.argv[1])
    Q_threshold = float(sys.argv[2])
    savename = 'test.pkl'

    joystick = Joystick()
    qnetwork = QNetwork(state_size=8, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load('basic_lander.pth'))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=0)

    episodes = 5
    scores = []
    data = []
    env.start_state(fx_init, 0.0)

    for episode in range(episodes):

        state = env.reset()
        env.render()
        score = 0

        while True:

            action, start, stop = joystick.input()
            if start:
                break

        while True:


            action, start, stop = joystick.input()
            data.append(list(state) + [action])

            with torch.no_grad():
                state = torch.from_numpy(state).float()
                Q_values = qnetwork(state).data.numpy()
            action_star = np.argmax(Q_values)
            loss = Q_values[action_star] - Q_values[action]
            if loss > Q_threshold:
                action = action_star

            env.render()
            state, reward, done, _ = env.step(action)
            score += reward

            if done or stop:
                print(episode, score)
                pickle.dump(data, open(savename, "wb" ))
                break
            time.sleep(0.025)

        scores.append(score)

    env.close()
    print(scores)


if __name__ == "__main__":
    main()
