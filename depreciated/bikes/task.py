import pygame
import sys
import numpy as np
import pickle
import torch
import math
from clone import MLP
from selfplay import MLP_MLP
from influence import I_MLP_MLP



class Model(object):

    def __init__(self):
        self.model = I_MLP_MLP()
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
        start = self.gamepad.get_button(0)
        stop = self.gamepad.get_button(7)
        return np.array([z1, z2]), start, stop


def net_display(screen, f, x, y, thickness=5, trirad=8):

    x = (x * 600) + 100
    y = (y * 600) + 100
    f *= 100.0

    start = [x, y]
    fx = f[0]
    fy = f[1]
    end = [0,0]
    end[0] = start[0]+fx
    end[1] = start[1]+fy
    rad = np.pi/180

    lcolor = (255, 179, 128)
    tricolor = (255, 179, 128)
    pygame.draw.line(screen, lcolor, start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + np.pi/2
    pygame.draw.polygon(screen, tricolor, ((end[0] + trirad * np.sin(rotation),
                                        end[1] + trirad * np.cos(rotation)),
                                       (end[0] + trirad * np.sin(rotation - 120*rad),
                                        end[1] + trirad * np.cos(rotation - 120*rad)),
                                       (end[0] + trirad * np.sin(rotation + 120*rad),
                                        end[1] + trirad * np.cos(rotation + 120*rad))))


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
    model = Model()

    pygame.init()
    world = pygame.display.set_mode([800,800])
    clock = pygame.time.Clock()
    fps = 10

    start = [0.25, 0.25]
    goal = [0.75, 0.8]
    player = Player(start[0], start[1])
    target = Goal(goal[0], goal[1])
    joystick = Joystick()
    sprite_list = pygame.sprite.Group()
    sprite_list.add(player)
    sprite_list.add(target)

    world.fill((0,0,0))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)
    data = []

    while True:

        ah, start, stop = joystick.input()
        if start:
            break

    while True:

        # human
        ah, start, stop = joystick.input()
        data.append([player.x, player.y, target.x, target.y] + list(ah))
        print([player.x, player.y, target.x, target.y] + list(ah))
        if stop:
            pickle.dump(data, open(savename, "wb" ))
            print(data)
            pygame.quit(); sys.exit()

        # human model
        context = torch.FloatTensor([player.x, player.y, target.x, target.y])
        ah = model.model.human(context)
        # ah = torch.FloatTensor(ah)

        # robot
        s = torch.FloatTensor([player.x, player.y])
        ar = model.model.robot(torch.cat((s, ah), 0)).detach().numpy()
        ar *= 4.0
        # ar = ah

        # dynamics
        player.update(ar)

        # animate
        world.fill((0,0,0))
        sprite_list.draw(world)
        net_display(world, ah, player.x, player.y)
        pygame.display.flip()
        clock.tick(fps)


if __name__ == "__main__":
    main()
