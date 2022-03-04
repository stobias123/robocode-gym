from concurrent import futures
import time
import logging
import subprocess
import gym
from gym import spaces
from base import BaseRobocodeEnv
from lib.robocode_manager import RoboCodeManager
from lib.connection_manager import ConnectionManager
from random import randint
import logging
import time
import numpy
import json
import http.client
import io, base64
from PIL import Image


class RobocodeV2(BaseRobocodeEnv):
  def __init__(self):
    super(RobocodeV2, self).__init__()
    logging.info(f"[BaseRobocodeEnv Env] - Version 2.0"
    self.last_frame = None)
    self.port_number = randint(32768,65535)
    self.robo_manager = RoboCodeManager(self.port_number)
    self.robo_manager.start()
    self.connection_manager = ConnectionManager(port_number=self.port_number)

  
    def reset(self):
        self.r.remote_client.reset()
        obs = self.step(0)
        return obs[0]
        #return self.last_frame, obs['reward'], obs['done'], obs['info']


    def step(self, action):
        # Send our action
        obs = self.r.remote_client.step(action)
        episode_over = obs['done']
        # self.r.remote_client.writeImage(obs['observation'])
        self.last_frame = self.r.remote_client.obsAsNumpyArray(obs['observation'])
        return self.last_frame, obs['reward'], obs['done'], obs['info']

    def _get_reward(self):
        pass

    def render(self,mode='rgb_array'):
        return self.last_frame

