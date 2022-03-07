import gym
import logging
import gym_robocode
from stable_baselines3 import PPO

model = PPO('MlpPolicy', 'Robocode-v0', verbose=1, tensorboard_log='./tensorboard_logs').learn(5)