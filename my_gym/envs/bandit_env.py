import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
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

    def __init__(self, total=4, good=2):
        super(BanditEnv, self).__init__()
        self.total = total
        self.good = good
        self.action_space = spaces.MultiDiscrete([total])
        self.observation_space = spaces.MultiDiscrete([2 for x in range(total)])
        self.reset()

    def step(self, a):

        self.a_taken = int(a)
        self.reward = self.state[self.a_taken]
        if self.reward == 0:
            self.reward = -1.
        return [self.state, self.reward, True, {}]

    def reset(self, test=False):
        self.state = np.zeros(self.total)
        if not test:
            self.good_arms = random.sample(range(self.total - 1), self.good)
        else:
            self.good_arms = random.sample(range(self.total), self.good)
        self.state[self.good_arms] = 1
        self.state = self.state.tolist()
        return self.state

    def render(self):
        print("State: {}, Arm Pulled: {}".format(self.good_arms, self.a_taken))