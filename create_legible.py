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


class TrajOpt(object):

    def __init__(self, traj, goals):
        """ set hyperparameters """
        self.n_waypoints = len(traj)
        self.n_joints = len(traj[0])
        self.home = traj[0]
        self.goals = goals
        """ create initial trajectory """
        self.xi0 = np.asarray(traj)
        self.action_limit = 0.05


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
        # cost_total += 0.1 * dist2goal
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
        cons = [{'type': 'eq', 'fun': self.cardinal_cons},
        {'type': 'ineq', 'fun': self.action_constraint}]
        start_t = time.time()
        res = minimize(self.trajcost, self.xi0, method=method, constraints=cons)
        xi = res.x.reshape(self.n_waypoints,self.n_joints)
        return xi, res, time.time() - start_t

def main():
    folder = "data/demos/"
    savename = "legible.pkl"
    dataname = folder + savename

    traj = pickle.load(open(dataname, "rb"))
    traj = np.asarray(traj)
    # print(traj)
               
    
    fig, ax = plt.subplots()
    ax.plot(traj[:,0], traj[:,1])

    G1 = np.asarray([0.8, 0.3])
    G2 = np.asarray([0.8, 0.4])
    goals = [G1, G2]

    traj_opt =TrajOpt(traj, goals)
    
    traj, res, t = traj_opt.optimize()
    ax.plot(traj[:,0], traj[:,1])

    plt.show()




if __name__ == "__main__":
    main()