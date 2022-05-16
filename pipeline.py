import logging
import os
from argparse import ArgumentParser

import mlflow
import gym
import time
from random import randint
from datetime import datetime
import boto3
from mlflow.entities import experiment
from mlflow.tracking import MlflowClient
import zipfile

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


def record_video(model: str , env_id: str , policy_name: str, record_steps: int, video_folder: str = 'artifacts/'):
    video_length = record_steps

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


env_id = 'RobocodeDownSample-v2'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--timesteps", dest="timesteps", help="timesteps to run training for", default=10, type=int)
    parser.add_argument("--record-timesteps", dest="record_timesteps", help="timesteps to run training for", default=10, type=int)
    parser.add_argument("--policy", dest="policy", help="policy", type=str)
    parser.add_argument("--load-model", dest="load_model", help="stable baselines 3 model path to load - no .zip", default='', type=str)
    args = parser.parse_args()

    client = MlflowClient()
    mlflow.set_tracking_uri("http://mlflow.10.20.50.162.nip.io:8080/")

    # foo
    runid = os.environ.get('MLFLOW_RUN_ID')
    print(runid)
    if runid is None:
        experiment = mlflow.set_experiment(experiment_name="robocode-example-2")
        runid = client.create_run(experiment_id=experiment.experiment_id).info.run_id


    with mlflow.start_run(run_id=runid) as active_run:
        policy = args.policy
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri - {artifact_uri}")
        mlflow.pytorch.autolog()
        mlflow.log_params({
            "env": env_id,
            "model": PPO,
            "policy": policy,
            "timesteps": args.timesteps
        })

        ## Train - should take ~23 mins at 7FPS.
        env = gym.make(env_id)
        logger = MLFlowLogger(folder='', output_formats="")
        if args.load_model == '':
            model = PPO(policy, env, verbose=1)
        else:
            logging.info(f"[Pipeline] Loading model.... {args.load_model}")
            model = PPO.load(args.load_model, env=env)
        model.set_logger(logger)


        logger.log(f"timesteps - {args.timesteps}")
        logger.log(f"record timesteps - {args.record_timesteps}")

        model.learn(total_timesteps=args.timesteps)
        model.save(f"/artifacts/models/{active_run.info.run_id}/robocode-model")

        ## Eval and record
        record_video(model, env_id, policy, video_folder=f"/artifacts/videos/{active_run.info.run_id}", record_steps=args.record_timesteps)
        #/mlflow/projects/code/videos/MlpPolicy-RobocodeDownSample-v2-step-0-to-step-1000.mp4
        uri = mlflow.get_artifact_uri()


        s3_client = boto3.client('s3')
        walks = os.walk('/artifacts/videos')

        for source, dirs, files in walks:
            for filename in files:
                local_file = os.path.join(source, filename)
                s3_client.upload_file(local_file,
                                      "mlflow-s3-bucket",
                                      f"videos/{local_file[1:]}")
        mlflow.log_artifacts('/artifacts/videos')
        print(f"Trying to find the mlflow videos from my current dir.")