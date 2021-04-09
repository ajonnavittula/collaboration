from gym.envs.registration import register

register(
    id='bandit-v0',
    entry_point='my_gym.envs:BanditEnv',
)
