import gym
import torch
import argparse
import yaml
import numpy as np

from hpbucrl import HPbUCRL

import logging
import datetime

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"log_file_{timestamp}.log"

    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-swimmer.yaml', help='config file')
    args = parser.parse_args()

    # load config file
    with open('configs/' + args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # define environment
    env = gym.make(config['env_name'])

    # initialize agent
    agent = HPbUCRL(env, config, device=device)

    # train agent
    agent.train()