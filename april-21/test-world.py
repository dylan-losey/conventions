import pygame
import sys
import os
import math
import numpy as np
import time
import random
import pickle
import copy
import torch
from models import Convention


class Joystick(object):

    def __init__(self):
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1

    def input(self):
        pygame.event.get()
        z = self.gamepad.get_axis(0)
        if abs(z) < self.DEADBAND:
            z = 0.0
        e_stop = self.gamepad.get_button(0)
        return z, e_stop


class Goal(pygame.sprite.Sprite):

    def __init__(self, x):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((25,25))
        self.image.fill((128, 128, 128))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.rect.x = (self.x * 1000) + 200 - self.rect.size[0] / 2
        self.rect.y = 100 - self.rect.size[1] / 2


class Player(pygame.sprite.Sprite):

    def __init__(self):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50,50))
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = 0
        self.xdot = 0.0
        self.rect.x = (self.x * 1000) + 200 - self.rect.size[0] / 2
        self.rect.y = 100 - self.rect.size[1] / 2

        # current inputs
        self.a = 0.0
        self.z = 0.0
        self.timestep = 0.05

    def update(self, a):

        # get the table
        self.rect = self.image.get_rect(center=self.rect.center)

        # integrate to get new state
        self.x = self.x + self.xdot * self.timestep
        self.xdot = self.xdot + float(a) * self.timestep

        # update the table position
        self.rect.x = (self.x * 1000) + 200 - self.rect.size[0] / 2


class Model(object):

    def __init__(self):
        self.model = Convention()
        model_dict = torch.load('models/influence_a1.pt', map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def human_initial(self, s_star, s):
        return self.model.human_initial(s_star, s)

    def human(self, s_star, s):
        return self.model.human(s_star, s)

    def robot(self, s, z):
        return self.model.robot(s, z)


def main():

    clock = pygame.time.Clock()
    pygame.init()
    fps = 20

    world = pygame.display.set_mode([1400,200])

    player = Player()
    joystick = Joystick()
    model = Model()

    g1 = Goal(0.2)
    g2 = Goal(0.4)
    g3 = Goal(0.6)
    g4 = Goal(0.8)
    g5 = Goal(1.0)

    sprite_list = pygame.sprite.Group()
    sprite_list.add(player)
    sprite_list.add(g1)
    sprite_list.add(g2)
    sprite_list.add(g3)
    sprite_list.add(g4)
    sprite_list.add(g5)

    while True:

        s_star = [0.8, 0.0]
        s = [player.x, player.xdot]
        print(s)

        # real human
        z, e_stop = joystick.input()
        z = torch.FloatTensor([z])
        if e_stop:
          pygame.quit(); sys.exit()

        # model human
        z = model.human(s_star, s)

        # robot
        a = model.robot(s, z)

        # dynamics
        player.update(a)

        # animate
        world.fill((0,0,0))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)


if __name__ == "__main__":
    main()
