import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import gym_robocode
from datetime import datetime


from stable_baselines3 import A2C


env_id = 'RobocodeDownSample-v2'
video_folder = 'logs/videos/'
video_length = 100

env = gym.make(env_id)
# Record the video starting at the first step
#env = VecVideoRecorder(env, video_folder,
#                       record_video_trigger=lambda x: x == 0, video_length=video_length,
#                       name_prefix=f"random-agent-{env_id}")

#model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=10000)

start = datetime.now()
obs = env.reset()
for i in range(100):
    if i % 250 == 0:
        now = datetime.now()
        delta = (now - start).total_seconds()
        print(f"Steps {i} in {delta}")
        print(f"Steps per second - {i/delta}")
    #action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(env.action_space.sample())
    #env.render()
    if done:
      obs = env.reset()
env.close()
