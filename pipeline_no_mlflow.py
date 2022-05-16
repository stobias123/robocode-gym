import logging
import os
from argparse import ArgumentParser

import gym
import time
from random import randint
from datetime import datetime
import string
import random
import boto3
import zipfile

import gym_robocode
from gym_robocode.envs.lib.connection_manager import ConnectionManager
from gym_robocode.envs.lib.k8s_manager import K8sManager
from gym_robocode.envs.lib.robocode_manager import RobocodeManagerImpl
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import Logger
from typing import Any, Optional, Tuple, Union
import logging
from collections import defaultdict

import kfp

kfp.Client()



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
    parser.add_argument("--policy", dest="policy", help="policy", default='CnnPolicy', type=str)
    parser.add_argument("--load-model", dest="load_model", help="stable baselines 3 model path to load - no .zip", default='', type=str)
    args = parser.parse_args()
    policy = args.policy
    ## Train - should take ~23 mins at 7FPS.
    env = gym.make(env_id)
    if args.load_model == '':
        model = PPO(policy, env, verbose=1)
    else:
        logging.info(f"[Pipeline] Loading model.... {args.load_model}")
        model = PPO.load(args.load_model, env=env)

    logging.info(f"timesteps - {args.timesteps}")
    logging.info(f"record timesteps - {args.record_timesteps}")

    model.learn(total_timesteps=args.timesteps)
    model.save(f"/artifacts/models/robocode-model")

    ## Eval and record
    record_video(model, env_id, policy, video_folder=f"/artifacts/videos/{run_id}", record_steps=args.record_timesteps)
    #/mlflow/projects/code/videos/MlpPolicy-RobocodeDownSample-v2-step-0-to-step-1000.mp4


    s3_client = boto3.client('s3')
    walks = os.walk('/artifacts/videos')

    for source, dirs, files in walks:
        for filename in files:
            local_file = os.path.join(source, filename)
            s3_client.upload_file(local_file,
                                  "mlflow-s3-bucket",
                                  f"videos/{local_file[1:]}")
    print(f"Trying to find the mlflow videos from my current dir.")