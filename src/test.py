import gymnasium as gym
import numpy as np
import argparse
import stable_baselines3 as sb3
import yaml

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from gymnasium.wrappers.record_video import RecordVideo


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='swimmer.yaml', help='config file')
    args = parser.parse_args()

    # load config file
    with open('configs/' + args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env_name = config["env_name"]
    env = RecordVideo(gym.make(env_name, render_mode="rgb_array"), video_length=1000, video_folder="videos/", episode_trigger=lambda x: x == 0, name_prefix=env_name)
    model = sb3.PPO.load("models/" + env_name)

    rewards = []

    for i in range(10):
        
        obs, _ = env.reset()
        cumulative_reward = 0

        for t in range(1000):

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action[:env.action_space.shape[0]])
            cumulative_reward += reward
            env.render()

            if terminated:
                break

        rewards.append(cumulative_reward)

    print("Average reward: ", np.mean(rewards))
    print("Standard deviation reward: ", np.std(rewards))

    env.close()