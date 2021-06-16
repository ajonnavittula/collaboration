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

class Robot(object):

    def __init__(self, home, goals):
        """ set hyperparameters """
        self.home = home
        # Set of all possible goals
        self.goalset = goals
        self.n_goals = len(goals)
        # current predicted goal
        self.goal_idx = None
        self.curr_pos = home
        # Number of possible robot and human actions
        self.r_actions = 30
        self.h_actions = 8
        # Action limits for robot and human
        self.r_limit = 0.05
        self.h_limit = 0.1
        # Action sets for robot and human
        self.r_actionset = None
        self.h_actionset = None
        self.create_actionsets()
        # Hyperparameter - human rationality for Boltzmann
        self.beta = 1.0
        # Plot contour map
        self.plot = False

    """ Create discrete action set for robot """
    def create_actionsets(self):
        angles = np.linspace(0, 2 * np.pi, self.r_actions)
        self.r_actionset = np.asarray([self.r_limit * np.cos(angles),\
                            self.r_limit * np.sin(angles) ])\
                            .reshape(self.r_actions, 2)
        angles = np.linspace(0, 2 * np.pi, self.h_actions)
        self.h_actionset = np.asarray([self.h_limit * np.cos(angles),\
                            self.h_limit * np.sin(angles) ])\
                            .reshape(self.h_actions, 2)

    """ Predict human's goal given current position """
    def predict(self, pos):
        dists = np.exp(np.linalg.norm(self.goalset - pos, axis=1))
        return np.argmin(dists)

    """ Run Bayes for a given action """
    def bayes(self, pos, a, agent="robot"):
        beta = 100.
        P = []
        for g in self.goalset:
            num = np.exp(-beta * np.linalg.norm(g - (pos + a)))
            den = 0
            # Computing bayes for human or robot actions?
            if agent == "robot":
                actionset = self.r_actionset
            else:
                actionset = self.h_actionset
                
            for ap in actionset:
                den += np.exp(-beta * np.linalg.norm(g - (pos + ap)))
            P.append(num / den)
        P = np.asarray(P)
        return P / sum(P)


    """ Get best robot action for given goal """
    def robot_action(self, pos):
        if not self.plot:
            self.goal_idx = self.predict(pos)
        start2goal = np.linalg.norm(self.goalset[self.goal_idx] - self.home)
        dist2goal = np.linalg.norm(self.goalset[self.goal_idx] - pos)
        # print(self.goal_idx)
        max_b = 0
        min_dist = np.inf
        best_action = np.asarray([0, 0])
        legible_action = np.asarray([0, 0])
        dist_action = np.asarray([0, 0])
        for a in self.actionset:
            belief = self.bayes(pos, a)
            dist = np.linalg.norm(self.goalset[self.goal_idx] - (pos + a))
            if dist < min_dist:
                dist_action = a
                min_dist = dist
            if belief[self.goal_idx] >= max_b:
                legible_action = a
                max_b = belief[self.goal_idx]
        alpha = np.clip(dist2goal / start2goal, 0., 1.0)
        best_action = alpha * legible_action + (1 - alpha) * dist_action
        return best_action

    """ Plot legible trajectories """
    def plot_arrow(self):
        if self.plot:
            self.goal_idx = 1
        print(self.goalset[self.goal_idx])
        fig, ax = plt.subplots()
        sx = np.linspace(-1, 1.0, 11)
        sy = np.linspace(-1, 1.0, 11)
        for x in sx:
            for y in sy:
                s = np.array([x, y])
                astar = self.robot_action(s)
                ax.arrow(x, y, astar[0], astar[1], head_width=0.1)
        
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
    postition_blue = np.asarray([0.8, 0.4])
    postition_green = np.asarray([0.8, 0.3])
    postition_gray = np.asarray([0., 0.])
    obs_position = postition_blue.tolist() + postition_green.tolist() + postition_gray.tolist()

    opt = TrajOpt(position_player, [postition_green, postition_blue])
    P_gain = 0.2

    opt.plot = False
    if opt.plot:
        opt.plot_arrow()
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
            opt.robot_action(q)
            # Get robot action only if human acts
            if not sum(a_h) == 0:
                a_r, conf = opt.robot_action(q)
                # a_r = np.clip(a_r, -0.05, 0.05)
            # print(a_r)

            if stop:
                pickle.dump( demonstration, open( savename, "wb" ) )
                print(demonstration)
                print("[*] Done!")
                print("[*] I recorded this many datapoints: ", len(demonstration))
                pygame.quit(); sys.exit()

            action = a_h + P_gain * a_r
            # print("Confidence: {0:2f}".format(conf))
            # action[1] =  (1 - conf) * a_h[1] + P_gain * conf * a_r[1]
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
