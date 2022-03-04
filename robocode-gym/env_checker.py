import gym
import logging
import gym_robocode
from stable_baselines3.common.env_checker import check_env

#env = OneOnOneEnv()
env = gym.make('Robocode-v2')
check_env(env)
