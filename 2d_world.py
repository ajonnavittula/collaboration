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
        self.h_actions = 30
        # Action limits for robot and human
        self.r_limit = 0.005
        self.h_limit = 0.01
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
        r = self.r_limit
        self.r_actionset = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        self.r_actionset = np.vstack((self.r_actionset, np.array([0.,  0.])))
        angles = np.linspace(0, 2 * np.pi, self.h_actions)
        r = self.h_limit
        self.h_actionset = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        self.h_actionset = np.vstack((self.h_actionset, np.array([0.,  0.])))

    """ Predict human's goal given current position """
    def predict(self, pos):
        dists = np.exp(np.linalg.norm(self.goalset - pos, axis=1))
        return np.argmin(dists)

    """ Run Bayes for a given action """
    def bayes(self, pos, a, agent="robot"):
        beta = 1000.
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
    def robot_action(self, pos, a_h):
        eps = 0.01
        if not self.plot:
            h_belief = self.bayes(pos, a_h, agent="human")
        self.goal_idx = 1
        # print("|a_h|: {}, conf: {}".\
        #         format(np.linalg.norm(a_h), belief[self.goal_idx]))
        max_b = 0
        min_dist = np.inf
        best_action = np.asarray([0, 0])
        legible_action = np.asarray([0, 0])
        dist_action = np.asarray([0, 0])
        Qmax = -np.Inf
        Q = {}
        for a in self.r_actionset:
            Q[str(a)] = np.linalg.norm(self.goalset[self.goal_idx] - pos) - \
                        np.linalg.norm(self.goalset[self.goal_idx] - (pos + a))
            if Q[str(a)] > Qmax:
                Qmax = Q[str(a)]
        for a in self.r_actionset:
            belief = self.bayes(pos, a)
            # dist = np.linalg.norm(self.goalset[self.goal_idx] - (pos + a))
            # if dist < min_dist:
            #     dist_action = a
            #     min_dist = dist
            if belief[self.goal_idx] >= max_b and Qmax - Q[str(a)] < eps:
                legible_action = a
                max_b = belief[self.goal_idx]
        # best_action = alpha * legible_action + (1 - alpha) * dist_action
        best_action = legible_action
        return best_action, h_belief[self.goal_idx]

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


def human_input(state, goals, goal_idx, angle):
    v = goals[goal_idx] - state
    unit_vec = v / np.linalg.norm(v)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],\
                        [np.sin(angle), np.cos(angle)]])
    rotated_vec = np.dot(rot_mat, unit_vec)
    return rotated_vec * 0.01 + np.random.normal(0, 0.005, 2)

def main():

    filename = sys.argv[1]
    savename = "data/demos/" + filename + ".pkl"

    position_player = np.asarray([.1, .35])
    postition_blue = np.asarray([0.8, 0.4])
    postition_green = np.asarray([0.8, 0.3])
    postition_gray = np.asarray([0., 0.])
    obs_position = postition_blue.tolist() + postition_green.tolist() + postition_gray.tolist()

    opt = Robot(position_player, [postition_green, postition_blue])
    P_gain = 0.2

    opt.plot = False
    if opt.plot:
        opt.plot_arrow()
    else:
        clock = pygame.time.Clock()
        pygame.init()
        fps = 30

        # world = pygame.display.set_mode([700, 700])

        player = Player(position_player)
        blue = Object(postition_blue, [0, 0, 255])
        green = Object(postition_green, [0, 255, 0])
        gray = Object(postition_gray, [128, 128, 128])

        # sprite_list = pygame.sprite.Group()
        # sprite_list.add(player)
        # sprite_list.add(blue)
        # sprite_list.add(green)
        # sprite_list.add(gray)

        # world.fill((0,0,0))
        # sprite_list.draw(world)
        # pygame.display.flip()
        clock.tick(fps)

        demonstration = []
        sampletime = 5
        count = 0
        t = 0
        # angle = np.random.uniform(-np.pi/2, np.pi/2)
        angle = 60 * np.pi / 180
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

            a_h = human_input(q, [postition_green, postition_blue], 1, angle)
            conf = 0.
            a_r = np.asarray([0, 0])

            # Get robot action only if human acts
            if not sum(a_h) == 0:
                a_r, conf = opt.robot_action(q, a_h)


            if stop:
                pickle.dump( demonstration, open( savename, "wb" ) )
                print(demonstration)
                print("[*] Done!")
                print("[*] I recorded this many datapoints: ", len(demonstration))
                pygame.quit(); sys.exit()

            P_gain = 0.
            action = a_h #+ P_gain * a_r
            # print("Confidence: {0:2f}".format(conf))
            q += action

            # dynamics
            player.update(q)

            # animate
            # world.fill((0,0,0))
            # sprite_list.draw(world)
            # pygame.display.flip()
            # clock.tick(fps)

            # pygame.event.pump()
            # save
            if not count % sampletime:
                demonstration.append(s)
            count += 1

            if count > 2:
                return [angle, conf]


if __name__ == "__main__":
    data = []
    filename = sys.argv[1]
    savename = "data/demos/" + filename + ".pkl"
    for i in range(250):
        res = main()
        data.append(res)
        print("iter: {0}, theta: {1:2f}, conf: {2:2f}".format(i, res[0], res[1]))
    print(data)
    pickle.dump( data, open( savename, "wb" ) )

    

