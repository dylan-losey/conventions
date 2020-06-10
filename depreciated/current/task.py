import pygame
import sys
import numpy as np
import pickle
import torch
from respond import R_MLP
from selfplay import MLP_MLP
from ideal import STAR_MLP
from influence import I_MLP



class Model(object):

    def __init__(self):
        self.model = I_MLP()
        model_dict = torch.load(self.model.name, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval


class Joystick(object):

    def __init__(self):
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        if abs(z1) < self.DEADBAND:
            z1 = 0.0
        if abs(z2) < self.DEADBAND:
            z2 = 0.0
        e_stop = self.gamepad.get_button(7)
        return np.array([z1, z2]), e_stop


class Goal(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((25,25))
        self.image.fill((128, 128, 128))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 600) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 600) + 100 - self.rect.size[1] / 2


class Player(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50,50))
        self.image.fill((255, 128, 0))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 600) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 600) + 100 - self.rect.size[1] / 2

    def update(self, a):

        # get the table
        self.rect = self.image.get_rect(center=self.rect.center)

        # integrate to get new state
        self.x += 0.01 * a[0]
        self.y += 0.01 * a[1]

        # update the table position
        self.rect.x = (self.x * 600) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 600) + 100 - self.rect.size[1] / 2



def main():

    n_test = sys.argv[1]
    savename = 'tests/test' + n_test + '.pkl'

    pygame.init()
    world = pygame.display.set_mode([800,800])
    clock = pygame.time.Clock()
    fps = 10

    start = np.random.random(2)
    player = Player(start[0], start[1])
    joystick = Joystick()
    model = Model()
    g1 = Goal(1.0, 0.0)
    g2 = Goal(0.0, 1.0)
    sprite_list = pygame.sprite.Group()
    sprite_list.add(player)
    sprite_list.add(g1)
    sprite_list.add(g2)

    world.fill((0,0,0))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)
    data = []


    while True:

        # human
        ah, e_stop = joystick.input()
        data.append([player.x, player.y, g1.x, g1.y, g2.x, g2.y] + list(ah))
        if e_stop:
            pickle.dump(data, open(savename, "wb" ))
            print(data)
            pygame.quit(); sys.exit()

        # robot
        s = torch.FloatTensor([player.x, player.y])
        ah = torch.FloatTensor(ah)
        context = torch.cat((s, ah), 0)
        ar = model.model.policy(context).detach().numpy()
        # ar = ah

        # dynamics
        player.update(ar)

        # animate
        world.fill((0,0,0))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)


if __name__ == "__main__":
    main()
