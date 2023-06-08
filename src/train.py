import gym
import torch
import argparse
import yaml
import wandb
import numpy as np

from hip_rl import HIPRL
from hallucination_wrapper import HallucinationWrapper

import logging
import datetime

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='swimmer.yaml', help='config file')
    args = parser.parse_args()

    # load config file
    with open('configs/' + args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # wandb support
    wandb.init(project="hip-rl", config=config)

    # set up logging
    timestap = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "logs/" + config['env_name'] + "_" + timestamp + ".txt"

    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # set up environment
    env = gym.make(config['env_name'])
    # env = HallucinationWrapper(env)
    # env = ClipReward(env, -1, 1)

    # initialize agent
    agent = HIPRL(env, config)

    # train agent
    agent.train()

    wandb.finish()