import pygame
import sys
import os
import math
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import pickle
import matplotlib.pyplot as plt


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

    def __init__(self, home, goals):
        """ set hyperparameters """
        self.n_waypoints = 30
        self.n_joints = 2
        self.home = home
        # Set of all possible goals
        self.goalset = goals
        # current predicted goal
        self.goal_idx = None
        self.xi0 = np.zeros((len(goals), self.n_waypoints, self.n_joints))
        self.action_limit = 0.05
        for waypoint in range(self.n_waypoints):
            for goal in range(len(goals)):
                self.xi0[goal, waypoint, :] = self.home + waypoint /(self.n_waypoints - 1.0)\
                                                 * (self.goalset[goal] - self.home) 
        self.optimize()

    """ Find most likely goal using Bayesian Inference """
    def predict(self, pos):
        dists = np.linalg.norm(self.goalset - pos, axis=1)
        idx = np.argmax(dists)
        print(idx)
        return idx, dists[idx]/sum(dists)

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
            costs = []
            for i in range(len(self.goalset)):
                costs.append(-np.exp(np.linalg.norm(self.goalset[i] - point)))
            cost_state = costs[self.goal_idx] / np.sum(costs)
            cost_total += cost_state * cost_scale
            dist2goal += np.linalg.norm(self.goalset[self.goal_idx] - point)
            if cost_scale > 1:
                cost_scale = len(xi) - idx
            smoothcost_xi += np.linalg.norm(xi[idx,:] - xi[idx-1,:])**2
        # perhaps you have some overall trajectory cost
        cost_total += 15 * smoothcost_xi
        cost_total += 0.1 * dist2goal
        return cost_total

    """ limit the actions to the given action set for trajectory"""
    def action_cons(self, xi):
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

    def start_cons(self, xi):
        xi = xi.reshape(self.n_waypoints,self.n_joints)
        start = xi[0,:]
        end = xi[-1, :]
        return np.linalg.norm(start - self.home)

    def end_cons(self, xi):
        xi = xi.reshape(self.n_waypoints,self.n_joints)
        start = xi[0,:]
        end = xi[-1, :]
        return np.linalg.norm(end - self.goalset[self.goal_idx])

    """ use scipy optimizer to get optimal trajectory """
    def optimize(self, method='SLSQP'):

        cons = [{'type': 'eq', 'fun': self.start_cons},
        {'type': 'eq', 'fun': self.end_cons},
        {'type': 'ineq', 'fun': self.action_cons}]
        for idx in range(len(self.goalset)):
            print("Optimizing goal {}".format(idx))
            self.goal_idx = idx
            start_t = time.time()
            xi0 = self.xi0[idx, :, :]
            xi0.reshape(-1)
            res = minimize(self.trajcost, xi0, method=method, constraints=cons)
            xi = res.x.reshape(self.n_waypoints,self.n_joints)
            self.xi0[idx, :, :] = xi

    """ P control to nudge user towards legible trajectory """
    def robot_action(self, pos, t):
        idx, conf = self.predict(pos)
        try:
            req_pos = self.xi0[idx, t, :]
        except IndexError:
            req_pos = self.xi0[idx, self.n_waypoints-1, :]
        return req_pos - pos, conf


    """ Plot legible trajectories """
    def plot(self):
        fig, ax = plt.subplots()
        for i in range(len(self.goalset)):
            ax.plot(self.xi0[i, :, 0], self.xi0[i, :, 1])

        plt.show()

        
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

    position_player = np.asarray([.1, .35])
    postition_blue = np.asarray([0.8, 0.5])
    postition_green = np.asarray([0.8, 0.2])
    postition_gray = np.asarray([0., 0.])
    obs_position = postition_blue.tolist() + postition_green.tolist() + postition_gray.tolist()

    opt = TrajOpt(position_player, [postition_blue, postition_green])
    P_gain = 0.1

    plot = False
    if plot:
        opt.plot()
    else:
        clock = pygame.time.Clock()
        pygame.init()
        fps = 30

        world = pygame.display.set_mode([700, 700])

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
        t = 0
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

            a_h = 0.01 * np.asarray([z1, z2])
            conf = 0.
            a_r = np.asarray([0, 0])
            # Get robot action only if human acts
            if not sum(a_h) == 0:
                if count % 10:
                    t += 1
                a_r, conf = opt.robot_action(q, t)
                a_r = np.clip(a_r, -0.05, 0.05)

            if stop:
                pickle.dump( demonstration, open( savename, "wb" ) )
                print(demonstration)
                print("[*] Done!")
                print("[*] I recorded this many datapoints: ", len(demonstration))
                pygame.quit(); sys.exit()

            action = a_h + P_gain * a_r
            q += action

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
