import gym
import logging
import gym_robocode
import time

logger = logging.getLogger('')
logging.basicConfig()
logger.setLevel(logging.DEBUG)

env = gym.make('RobocodeDownSample-v2')

action = env.action_space.sample()
env.reset()
while(True):
    action = env.action_space.sample()
    env.step(action)
    time.sleep(1)
