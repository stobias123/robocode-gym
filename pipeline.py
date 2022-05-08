import os
from argparse import ArgumentParser

import mlflow
import gym
import time
import click
from random import randint
from datetime import datetime
from mlflow.entities import experiment
from mlflow.tracking import MlflowClient

import gym_robocode
from mlflow import log_metric, log_param, log_artifacts
from gym_robocode.envs.lib.connection_manager import ConnectionManager
from gym_robocode.envs.lib.k8s_manager import K8sManager
from gym_robocode.envs.lib.robocode_manager import RobocodeManagerImpl
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import Logger
from typing import Any, Optional, Tuple, Union
from collections import defaultdict





class MLFlowLogger(Logger):
    def init(self):
        super.__init__(folder='output', output_formats="json")
        self.name_to_value = defaultdict(float)  # values this iteration
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(str)
        self.level = 20

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        mlflow.log_metric(key, value)


def record_video(model: str , env_id: str , policy_name: str, record_steps: int):
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--timesteps", dest="timesteps", help="timesteps to run training for", default=None, type=int)
    parser.add_argument("--record-timesteps", dest="record-timesteps", help="timesteps to run training for", default=None, type=int)
    args = parser.parse_args()

    client = MlflowClient()
    mlflow.set_tracking_uri("http://mlflow.10.20.50.162.nip.io:8080/")

    #experiment = mlflow.set_experiment(experiment_name="robocode-example-2")

    with mlflow.start_run() as active_run:
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri - {artifact_uri}")
        mlflow.pytorch.autolog()
        mlflow.log_params({
            "env": env_id,
            "model": "PPO",
            "policy": policy,
            "timesteps": args.timesteps
        })

        ## Train - should take ~23 mins at 7FPS.
        env = gym.make(env_id)
        logger = MLFlowLogger(folder='', output_formats="")
        model = PPO(policy, env, verbose=1)
        model.set_logger(logger)
        model.learn(total_timesteps=args.timesteps)
        model.save('robocode-model')


        ## Eval and record
        record_video(model, env_id, policy, record_steps=10)
        #/mlflow/projects/code/videos/MlpPolicy-RobocodeDownSample-v2-step-0-to-step-1000.mp4
        uri = mlflow.get_artifact_uri()
        print(f"Trying to upload to {uri}.")
        mlflow.log_artifact('/mlflow/projects/code/robocode-model.zip')
        mlflow.log_artifacts('/mlflow/projects/code/models/')
        print(f"Trying to find the mlflow videos from my current dir.")