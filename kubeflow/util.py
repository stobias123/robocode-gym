from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import gym

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

