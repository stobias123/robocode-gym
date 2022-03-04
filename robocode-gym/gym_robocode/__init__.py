import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="Robocode-v0", entry_point="gym_robocode.envs:OneOnOneEnv")
register(id="Robocode-v2", entry_point="gym_robocode.envs:RobocodeV2")