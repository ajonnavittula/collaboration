import pygame
import sys
import os
import math
import numpy as np
import time
import pickle


class Joystick(object):

    def __init__(self):
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        if abs(z1) < self.DEADBAND:
            z1 = 0.0
        z2 = self.gamepad.get_axis(1)
        if abs(z2) < self.DEADBAND:
            z2 = 0.0
        start = self.gamepad.get_button(1)
        stop = self.gamepad.get_button(0)
        return [z1, z2], start, stop

class TrajOpt(object):

    def __init__(self, traj, goals):
        """ set hyperparameters """
        self.n_waypoints = len(traj)
        self.n_joints = len(traj[0])
        self.home = traj[0]
        # Set of all possible goals
        self.goalset = goals
        # current predicted goal
        self.goal_idx = None
        self.xi0 = np.asarray(traj)
        self.action_limit = 0.05

    """ Find most likely goal using Bayesian Inference """
    def bayes(self, pos):
        return np.argmax(np.linalg.norm(self.goalset - pos))

    """ problem specific cost function """
    def trajcost(self, xi):
        xi = xi.reshape(self.n_waypoints,self.n_joints)
        cost_total = 0
        smoothcost_xi = 0
        dist2goal = 0
        cost_scale = len(xi)
        for idx in range(1, self.n_waypoints):
            # state cost goes here
            point = xi[idx]
            cost_g1 = np.exp(-np.linalg.norm(self.goals[0] - point))
            cost_g2 = np.exp(-np.linalg.norm(self.goals[1] - point))
            cost_state = cost_g1 / (cost_g1 + cost_g2)
            cost_total += cost_state * cost_scale
            dist2goal += np.linalg.norm(self.goals[1] - point)
            if cost_scale > 1:
                cost_scale = len(xi) - idx
            smoothcost_xi += np.linalg.norm(xi[idx,:] - xi[idx-1,:])**2
        # perhaps you have some overall trajectory cost
        cost_total += 10 * smoothcost_xi
        cost_total += 0.1 * dist2goal
        return cost_total

    """ limit the actions to the given action set for trajectory"""
    def action_constraint(self, xi):
        xi = xi.reshape(self.n_waypoints,self.n_joints)
        length = 0
        prev_point = xi[0, :]
        max_diff = 0.
        for idx in range(1, len(xi)):
            point = xi[idx, :]
            diff = point - prev_point
            if max(diff) > max_diff:
                max_diff = max(diff)
            prev_point = point
                    
        return self.action_limit - max_diff

    def cardinal_cons(self, xi):
        xi = xi.reshape(self.n_waypoints,self.n_joints)
        start = xi[0,:]
        end = xi[-1, :]

        return np.linalg.norm(start - self.home) + \
                np.linalg.norm(end - self.goals[1])

    """ use scipy optimizer to get optimal trajectory """
    def optimize(self, method='SLSQP'):
        if self.goal is None:
            self.goal = goal[self.bayes()]
        else:
            idx = self.bayes()
            if not self.goal_idx == idx:
                self.goal_idx = idx

        cons = [{'type': 'eq', 'fun': self.cardinal_cons},
        {'type': 'ineq', 'fun': self.action_constraint}]
        start_t = time.time()
        res = minimize(self.trajcost, self.xi0, method=method, constraints=cons)
        xi = res.x.reshape(self.n_waypoints,self.n_joints)
        return xi, res, time.time() - start_t
        
class Object(pygame.sprite.Sprite):

    def __init__(self, position, color):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((25,25))
        self.image.fill(color)
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = (self.x * 500) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 500) + 100 - self.rect.size[1] / 2


class Player(pygame.sprite.Sprite):

    def __init__(self, position):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50,50))
        self.image.fill((255, 128, 0))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = (self.x * 500) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 500) + 100 - self.rect.size[1] / 2

    def update(self, s):
        self.rect = self.image.get_rect(center=self.rect.center)
        self.x = s[0]
        self.y = s[1]
        self.rect.x = (self.x * 500) + 100 - self.rect.size[0] / 2
        self.rect.y = (self.y * 500) + 100 - self.rect.size[1] / 2


def main():

    filename = sys.argv[1]
    savename = "data/demos/" + filename + ".pkl"

    clock = pygame.time.Clock()
    pygame.init()
    fps = 30

    world = pygame.display.set_mode([700, 700])
    position_player = np.asarray([.1, .35])
    postition_blue = np.asarray([0.8, 0.4])
    postition_green = np.asarray([0.8, 0.3])
    postition_gray = np.asarray([0., 0.])
    obs_position = postition_blue.tolist() + postition_green.tolist() + postition_gray.tolist()


    player = Player(position_player)
    blue = Object(postition_blue, [0, 0, 255])
    green = Object(postition_green, [0, 255, 0])
    gray = Object(postition_gray, [128, 128, 128])

    sprite_list = pygame.sprite.Group()
    sprite_list.add(player)
    sprite_list.add(blue)
    sprite_list.add(green)
    sprite_list.add(gray)

    world.fill((0,0,0))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)

    demonstration = []
    sampletime = 5
    count = 0

    while True:

        q = np.asarray([player.x, player.y])
        # s = obs_position + q.tolist()
        s = q.tolist()

        stop = 0
        z1 = 0
        z2 = 0
        action = [0, 0]
        left = pygame.key.get_pressed()[pygame.K_LEFT]
        up = pygame.key.get_pressed()[pygame.K_UP]
        down = pygame.key.get_pressed()[pygame.K_DOWN]
        right = pygame.key.get_pressed()[pygame.K_RIGHT]
        stop = pygame.key.get_pressed()[pygame.K_RETURN]
        
        z1 = right - left
        z2 = down - up

        action = [z1, z2]

        if stop:
            pickle.dump( demonstration, open( savename, "wb" ) )
            print(demonstration)
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            pygame.quit(); sys.exit()

        q += np.asarray(action) * 0.01

        # dynamics
        player.update(q)

        # animate
        world.fill((0,0,0))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)

        pygame.event.pump()
        # save
        if not count % sampletime:
            demonstration.append(s)
        count += 1

if __name__ == "__main__":
    main()
