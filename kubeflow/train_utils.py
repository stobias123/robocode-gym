import kfp.components
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)


@component
def train(timesteps: int, record_timesteps: int, load_model: str = '', policy: str = 'CnnPolicy'):
    import os
    import gym
    import gym_robocode
    import string
    import random
    import boto3
    from kubeflow.util import record_video
    from stable_baselines3 import A2C, PPO
    import logging

    env_id = 'RobocodeDownSample-v2'
    letters = string.ascii_lowercase
    run_id = ''.join(random.choice(letters) for i in range(10))
    ## Train - should take ~23 mins at 7FPS.
    env = gym.make(env_id)
    if load_model == '':
        model = PPO(policy, env, verbose=1)
    else:
        logging.info(f"[Pipeline] Loading model.... {load_model}")
        model = PPO.load(load_model, env=env)

    logging.info(f"timesteps - {timesteps}")
    logging.info(f"record timesteps - {record_timesteps}")

    model.learn(total_timesteps=timesteps)
    model.save(f"/artifacts/models/robocode-model")

    ## Eval and record
    record_video(model, env_id, policy, video_folder=f"/artifacts/videos/{run_id}", record_steps=record_timesteps)
    # /mlflow/projects/code/videos/MlpPolicy-RobocodeDownSample-v2-step-0-to-step-1000.mp4

    s3_client = boto3.client('s3')
    walks = os.walk('/artifacts/videos')

    for source, dirs, files in walks:
        for filename in files:
            local_file = os.path.join(source, filename)
            s3_client.upload_file(local_file,
                                  "mlflow-s3-bucket",
                                  f"videos/{local_file[1:]}")
    print(f"Trying to find the mlflow videos from my current dir.")