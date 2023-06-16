import numpy as np
import argparse
import yaml
import pandas as pd

import matplotlib.pyplot as plt


if __name__ == "__main__":

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='swimmer.yaml', help='config file')
    args = parser.parse_args()

    # load config file
    with open('configs/' + args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env_name = config["env_name"]

    # load second column of reward
    reward = pd.read_csv("data/" + env_name + "_reward.csv", skiprows=0).iloc[:, 1]
    # moving average where nan values are filled with the last valid observation
    reward = reward.rolling(window=10, min_periods=1).mean()

    # load second column of reward deviation
    reward_deviation = pd.read_csv("data/" + env_name + "_deviation.csv", skiprows=0).iloc[:, 1]

    # plot reward with deviation
    plt.figure(figsize=(8, 6))
    plt.plot(reward, label="average reward")
    plt.fill_between(reward.index, reward - reward_deviation, reward + reward_deviation, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Everage Reward")
    plt.legend(loc="upper left")
    plt.savefig("images/" + env_name + "_reward.pdf", bbox_inches='tight')
    plt.close()

    # load second column of transition deviation
    transition_deviation = pd.read_csv("data/" + env_name + "_transition.csv", skiprows=0).iloc[:, 1]
    
    # plot transition deviation
    plt.figure(figsize=(8, 6))
    plt.plot(transition_deviation, label="transition deviation")
    plt.xlabel("Episode")
    plt.ylabel("Transition Deviation")
    plt.legend()
    plt.savefig("images/" + env_name + "_transition.pdf", bbox_inches='tight')
    plt.close()

    # load second column of preference deviation
    reward_deviation = pd.read_csv("data/" + env_name + "_preference.csv", skiprows=0).iloc[:, 1]

    # plot reward deviation
    plt.figure(figsize=(8, 6))
    plt.plot(reward_deviation, label="preference deviation")
    plt.xlabel("Episode")
    plt.ylabel("Preference Deviation")
    plt.legend()
    plt.savefig("images/" + env_name + "_preference.pdf", bbox_inches='tight')
    plt.close()

