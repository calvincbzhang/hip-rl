import gym
import torch
import argparse
import yaml
import numpy as np

from hpbucrl import HPbUCRL

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-swimmer.yaml', help='config file')
    args = parser.parse_args()

    # load config file
    with open('configs/' + args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # define environment
    env = gym.make(config['env_name'])

    # initialize agent
    agent = HPbUCRL(env, config)

    # train agent
    agent.train()
