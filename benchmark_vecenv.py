import gym
import logging
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable
import gym_robocode
from stable_baselines3 import PPO
import time
from stable_baselines3.common.utils import set_random_seed

logger = logging.getLogger('')
logging.basicConfig()
logger.setLevel(logging.INFO)
env_id = 'RobocodeDownSample-v2'

env = gym.make(env_id)

action = env.action_space.sample()
env.reset()


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    n_timesteps = 2500

    # Multiprocessed RL Training
    env = SubprocVecEnv([make_env(env_id, i) for i in range(16)])
    model = PPO('MlpPolicy', env, verbose=0)
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")

    # Single Process RL Training
    single_process_model = PPO('MlpPolicy', env_id, verbose=0)

    start_time = time.time()
    single_process_model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print(f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS")

    print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))