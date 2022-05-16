import mlflow
import os
import boto3
import torch
from mlflow.entities import experiment
from mlflow.tracking import MlflowClient
from stable_baselines3 import PPO
from pipeline import record_video
from argparse import ArgumentParser

env_id = 'RobocodeDownSample-v2'
policy = 'CnnPolicy'

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", dest="model", help="model", type=str)
    args = parser.parse_args()

    client = MlflowClient()
    artifact_uri = mlflow.get_artifact_uri()
    print(artifact_uri)
    mlflow.set_tracking_uri("http://mlflow.10.20.50.162.nip.io:8080/")
    mlflow.set_experiment(experiment_name="robocode-example-2")
    run = client.create_run('1')

    model = PPO.load(args.model)
    record_video(model, env_id, policy, video_folder='/artifacts/videos', record_steps=1000)
    mlflow.log_artifacts('/artifacts/videos')

    s3_client = boto3.client('s3')
    walks = os.walk('/artifacts/videos')

    for source, dirs, files in walks:
        for filename in files:
            local_file = os.path.join(source, filename)
            s3_client.upload_file(local_file,
                                  "mlflow-s3-bucket",
                                  f"videos/{local_file}")
            mlflow.log_artifacts('/artifacts/videos')
            print(f"Trying to find the mlflow videos from my current dir.")

# print(type(model.policy.state_dict()))
# print(model.policy_class)


# mlflow.pytorch.log_model(model.policy.state_dict(), 'my_model')

# torch.load('model_unzipped/')
