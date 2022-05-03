import logging
from gym_robocode.envs.base import BaseRobocodeEnv
from gym_robocode.envs.lib.robocode_manager import RobocodeManager
from gym_robocode.envs.lib.connection_manager import ConnectionManager
from random import randint
from PIL import Image

logging.basicConfig(level=logging.INFO)


class RobocodeV2(BaseRobocodeEnv):
    """
    Robocodev2 environment spawns a new headless robocode for every instantiation.
    """

    def __init__(self):
        super(RobocodeV2, self).__init__()
        self.robo_manager: RobocodeManager = None
        self.connection_manager: ConnectionManager = None
        self.port_number: int = None
        logging.info(f"[BaseRobocodeEnv Env] - Version 2.0")
        self.last_frame = None
        self.episode_over = False

    ## robo manager must be started before this happens.
    def init(self, robo_manager: RobocodeManager, connection_manager: ConnectionManager):
        self.robo_manager = robo_manager
        self.port_number = self.robo_manager.port_number
        self.connection_manager = connection_manager

    def reset(self):
        self.connection_manager.reset()
        obs = self.step(0)
        return obs[0]

    def step(self, action):
        # Send our action
        obs = self.connection_manager.step(action)
        self.episode_over = obs['done']
        self.last_frame = self.connection_manager.obsAsNumpyArray(obs['observation'])
        return self.last_frame, obs['reward'], obs['done'], obs['info']

    def _get_reward(self):
        pass

    def render(self, mode='rgb_array'):
        return self.last_frame

    def close(self):
        self.robo_manager.stop()
