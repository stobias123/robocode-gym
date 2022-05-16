import kfp
import kfp.components as comp
import kfp.dsl as dsl
from kubeflow.train_utils import train


@dsl.pipeline( name="robocode pipeline")
def pipeline(timesteps: int, record_timesteps: int,policy: str ="CnnPolicy"):
    train(timesteps=timesteps, record_timesteps=record_timesteps, policy=policy)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, package_path='pipeline.yaml')