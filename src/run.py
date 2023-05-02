import gym
import torch
import numpy as np

from transition_model import GPTransition
from preference_model import GPPreference

from util import *
    

if __name__ == "__main__":
    # define environment
    env_1 = gym.make('InvertedPendulum-v4')
    env_2 = gym.make('InvertedPendulum-v4')
    state_dim = env_1.observation_space.shape
    action_dim = env_1.action_space.shape

    # some hyperparams
    HORIZON = 100
    EPISODES = 20

    # get state and action as torch tensors
    state = torch.zeros(1, state_dim[0])
    action = torch.zeros(1, action_dim[0])
    next_state = torch.zeros(1, state_dim[0])

    # transition model
    transition = GPTransition(state, action, next_state)
    optimizer = torch.optim.Adam(transition.parameters(), lr=0.1)

    # preference model
    preference = GPPreference(state, action, HORIZON)
    preference_optimizer = torch.optim.Adam(preference.parameters(), lr=0.1)

    preferences = {}
    trajectories = []

    transition.train()
    preference.train()

    for ep in range(EPISODES):

        ep_reward_1 = 0
        ep_reward_2 = 0

        obs_1 = env_1.reset()[0]
        obs_2 = env_2.reset()[0]

        t_1 = np.array([obs_1])
        t_2 = np.array([obs_2])

        for time in range(HORIZON):
            action_1 = env_1.action_space.sample()
            action_2 = env_2.action_space.sample()

            next_obs_1, reward_1, done_1, _, info_1 = env_1.step(action_1)
            ep_reward_1 += reward_1

            transition.add_data(torch.tensor(obs_1).unsqueeze(0),
                torch.tensor(action_1).unsqueeze(0),
                torch.tensor(next_obs_1).unsqueeze(0)
            )

            next_obs_2, reward_2, done_2, _, info_2 = env_2.step(action_2)
            ep_reward_2 += reward_2

            transition.add_data(torch.tensor(obs_2).unsqueeze(0),
                torch.tensor(action_2).unsqueeze(0),
                torch.tensor(next_obs_2).unsqueeze(0)
            )
            
            t_1 = np.append(t_1, action_1)
            t_2 = np.append(t_2, action_2)
            if time != HORIZON - 1:
                t_1 = np.append(t_1, next_obs_1)
                t_2 = np.append(t_2, next_obs_2)

            optimizer.zero_grad()

            output = tensor_to_distribution(
                transition(torch.tensor(obs_1).unsqueeze(0), torch.tensor(action_1).unsqueeze(0))
            )
            with gpytorch.settings.fast_pred_var():
                val = torch.stack(tuple([gp.train_targets for gp in transition.gp]), 0)
                loss = exact_mll(output, val, transition.gp)
                if time % 10 == 0:
                    print(loss)
            loss.backward()
            optimizer.step()

            output = tensor_to_distribution(
                transition(torch.tensor(obs_2).unsqueeze(0), torch.tensor(action_2).unsqueeze(0))
            )
            with gpytorch.settings.fast_pred_var():
                val = torch.stack(tuple([gp.train_targets for gp in transition.gp]), 0)
                loss = exact_mll(output, val, transition.gp)
                if time % 10 == 0:
                    print(loss)
            loss.backward()
            optimizer.step()

            obs_1 = next_obs_1
            obs_2 = next_obs_2

        # add data to preference model
        trajectories.append(t_1)
        trajectories.append(t_2)
        preference.add_data(t_1, t_2)

        # print("Episode: {}, Reward 1: {}, Rewards 2: {}".format(ep+1, ep_reward_1, ep_reward_2))

    print(preferences)