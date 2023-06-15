import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3

if __name__ == "__main__":

    env = gym.make("Swimmer-v4", render_mode="human")
    model = sb3.SAC.load("models/Swimmer-v4")
    obs, _ = env.reset()

    cum_r = 0

    for i in range(1000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, _, _, _ = env.step(action[:env.action_space.shape[0]])
        cum_r += rewards

    print("Cumulative reward: ", cum_r)
    
    # env.close()