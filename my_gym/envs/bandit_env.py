import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import sys

class BanditEnv(gym.Env):
    """
    A simple multi-arm bandit ennvironment for conventions. Creates an environment with 
    given number of arms and good arms. Returns reward = 1.0 if good arm pulled and -1
    otherwise.

    total: Total number of arms
    good: Number of good arms
    """

    def __init__(self, total, good):
        super(BanditEnv, self).__init__()
        self.total = total
        self.good = good
        self.action_space = spaces.MultiDiscrete([total, total])
        self.observation_space = spaces.MultiDiscrete([total])
        self.reset()

    def step(self, a):
        self.reward = self.state[a]
        return [self.state, self.reward, True, {}]

    def reset(self):
        self.state = np.ones(self.total) * -1
        self.good_arms = random.sample(range(self.total - 1), self.good)
        self.state[self.good_arms] = 1
        return self.state

    def render(self):
        print("State: {}, Good Arms: {}".format(self.good, self.state))