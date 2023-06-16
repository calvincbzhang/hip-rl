import gymnasium as gym
import torch
import argparse
import yaml
import wandb
import numpy as np

import logging
import datetime
import os

import random
import stable_baselines3 as sb3
from stable_baselines3 import PPO

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set a fixed seed
seed = 42 

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    foldername = "logs/" + config['env_name'] + "_" + timestamp
    os.mkdir(foldername)
    filename = foldername + "/log.txt"

    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # set up environment
    env = gym.make(config['env_name'])

    # set up agent
    if config['env_name'] == "HalfCheetah-v4" or config['env_name'] == "Hopper-v4":
        model = PPO(
            "MlpPolicy",
            verbose=1,
            env=env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            policy_kwargs={"log_std_init": -2, "ortho_init": False, "activation_fn": torch.nn.ReLU, "net_arch": [{"pi": [256, 256], "vf": [256, 256]}]},
        )
    else:
        model = PPO(
            "MlpPolicy",
            verbose=1,
            env=env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
        )

    # train model
    model.learn(total_timesteps=config['total_timesteps']*10, progress_bar=True)

    # save model
    model.save("models/PPO_" + config['env_name'])

    wandb.finish()