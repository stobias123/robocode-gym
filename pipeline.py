from random import randint

import mlflow
import gym
import gym_robocode
from gym_robocode.envs.lib.connection_manager import ConnectionManager
from gym_robocode.envs.lib.k8s_manager import K8sManager
from mlflow import log_metric, log_param, log_artifacts
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import Logger
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
from collections import defaultdict

from datetime import datetime

from gym_robocode.envs.lib.robocode_manager import RobocodeManagerImpl

mlflow.set_tracking_uri("http://mlflow.10.20.50.162.nip.io:8080/")
mlflow.set_experiment("robocode")
mlflow.pytorch.autolog()


class MLFlowLogger(Logger):
    def init(self):
        super.__init__(folder='output', output_formats="json")
        self.name_to_value = defaultdict(float)  # values this iteration
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(str)
        self.level = 20

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        mlflow.log_metric(key, value)


def record_video(model, env_id, policy_name):
    video_folder = 'videos/'
    video_length = 1000

    env = DummyVecEnv([lambda: gym.make(env_id)])

    obs = env.reset()
    # Record the video starting at the first step
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"{policy_name}" + "-{}".format(env_id))
    env.reset()
    for _ in range(video_length + 1):
        action, _states = model.predict(obs)
        obs, _, _, _ = env.step(action)
    # Save the video
    env.close()


policy = 'MlpPolicy'
env_id = 'RobocodeDownSample-v2'

with mlflow.start_run() as active_run:
    mlflow.log_params({
        "env": env_id,
        "model": "PPO",
        "policy": policy}
    )

    ## Train - should take ~23 mins at 7FPS.
    robo_manager = K8sManager(namespace='robocode')
    connection_manager = ConnectionManager(port_number=robo_manager.port_number)
    env = gym.make(env_id)
    env.init(robo_manager, connection_manager)
    logger = MLFlowLogger(folder='', output_formats="")
    model = PPO(policy, env, verbose=1)
    model.set_logger(logger)
    model.learn(total_timesteps=5000)

    ## Eval and record
    record_video(model, env_id, policy)
