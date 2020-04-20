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
from rnn_models import RNNAE
from inter_models import TEAM


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
        e_stop = self.gamepad.get_button(7)
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
        self.image.fill((255, 128, 0))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = 0.0
        self.rect.x = (self.x * 1000) + 200 - self.rect.size[0] / 2
        self.rect.y = 100 - self.rect.size[1] / 2

    def update(self, s):

        # get the table
        self.rect = self.image.get_rect(center=self.rect.center)

        # integrate to get new state
        self.x = s.item()

        # update the table position
        self.rect.x = (self.x * 1000) + 200 - self.rect.size[0] / 2


# class Model(object):
#
#     def __init__(self, modelname):
#         self.model = RNNAE()
#         model_dict = torch.load(modelname, map_location='cpu')
#         self.model.load_state_dict(model_dict)
#         self.model.eval
#
#     def robot(self, input, hidden):
#         output, hidden = self.model.robot(input, hidden)
#         return output, hidden


class Model(object):

    def __init__(self, modelname):
        self.model = TEAM()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def robot(self, input, hidden):
        output, hidden = self.model.R(input, hidden)
        return output, hidden


def main():

    clock = pygame.time.Clock()
    pygame.init()
    fps = 2

    world = pygame.display.set_mode([1400,200])
    modelname = 'models/team-1.pt'

    player = Player()
    joystick = Joystick()
    model = Model(modelname)

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

    count = 0
    hidden = model.model.init_hidden()

    world.fill((0,0,0))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)

    while count < 10:

        s = torch.tensor(player.x)
        s = s.view(1)

        # real human
        ah, e_stop = joystick.input()
        ah = torch.tensor(ah)
        if e_stop:
            pygame.quit(); sys.exit()

        context = torch.cat((s, ah.view(1)), 0)

        # robot
        ar, hidden = model.robot(context, hidden)
        s = s + ar

        # dynamics
        player.update(s)

        # animate
        world.fill((0,0,0))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)
        count += 1
        print(count, ah)


if __name__ == "__main__":
    main()
