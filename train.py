import gym
import numpy as np
import my_gym

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "bandit-v0"
    num_cpu = 4  # Number of processes to use
    # # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=[10, 5])
    # env = gym.make(env_id, total=10, good=3)
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=15000)

    env = gym.make(env_id)#, total=10, good=3)
    for _ in range(10):
        obs = env.reset(test=True)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()