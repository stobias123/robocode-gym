import gym
from gym import wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym_robocode

env_id = 'Robocode-v0'
video_folder = 'logs/videos/'
video_length = 100

env= gym.make(env_id)
rec = VideoRecorder(env,'foo.mp4',enabled=True)
obs = env.reset()
rec.capture_frame()

while(not obs[2]):
  env.render()
  rec.capture_frame()
  obs = env.step(env.action_space.sample())

print("saved video")
rec.close()