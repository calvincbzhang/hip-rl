import numpy as np
import torch

from transition_model import GPTransitionModel
from reward_model import RewardModel
from policy import Policy

import logging


class HPbUCRL:
    def __init__(self, env, config, device='cpu'):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.device = device

        # TODO: initialize parameters
        self.horizon = config['horizon']
        self.train_epochs = config['train_epochs']

        state = torch.zeros(1, self.state_dim, device=device)
        action = torch.zeros(1, self.action_dim, device=device)
        next_state = torch.zeros(1, self.state_dim, device=device)

        # initialize transition model
        self.transition_model = GPTransitionModel(state, action, next_state, device=device)
        # initialize reward model
        self.reward_model = RewardModel(self.state_dim, self.action_dim)
        # initialize policy
        self.policy = Policy(self.state_dim, self.action_dim)

        self.T = []
        self.R = []
        self.P = []
    
    def train(self):
        # append random initial trajectory to T
        s = self.env.reset()[0]
        tau = [s]
        reward = 0
        for _ in range(self.horizon):
            a = self.env.action_space.sample()
            s_next, r, _, _, _ = self.env.step(a)

            # add data to the transition model
            self.transition_model.add_data(torch.tensor(s, device=self.device),
                                           torch.tensor(a, device=self.device),
                                           torch.tensor(s_next, device=self.device))

            s = s_next
            reward += r
            tau.append(a)
            tau.append(s_next)

        self.T.append(tau)
        self.R.append(reward)

        # train for n_episodes
        for k in range(self.train_epochs):
            s = self.env.reset()[0]
            tau = [s]
            reward = 0
            for t in range(self.horizon):
                # sample action from policy and to plan step in algorithm
                a = self.policy.sample_action(s)
                s_next, r, _, _, _ = self.env.step(a)

                # add data to the transition model
                self.transition_model.add_data(torch.tensor(s, device=self.device),
                                               torch.tensor(a, device=self.device),
                                               torch.tensor(s_next, device=self.device))

                s = s_next
                reward += r
                tau.append(a)
                tau.append(s_next)

            # sample old trajectory and its reward
            idx = np.random.randint(len(self.T))
            tau_old = self.T[idx]
            reward_old = self.R[idx]

            self.T.append(tau)
            self.R.append(reward)

            # log data
            print(f'Episode {k} - Reward: {reward} - Old Reward: {reward_old}')
            print(f'Episode {k} - Rewards: {self.R}')
            logging.info(f'Episode {k} - Reward: {reward} - Old Reward: {reward_old}')
            logging.info(f'Episode {k} - Rewards: {self.R}')

            # TODO: add stochasticity to the preference
            # compute binary preference between new and old trajectory
            if reward > reward_old:
                self.P.append([tau, tau_old, 1])
            else:
                self.P.append([tau_old, tau, 0])

            # estimate reward
            self.reward_model.train(self.P)
            # estimate transition model
            self.transition_model.train(verbose=True)
            # train policy
            self.policy.train_policy(self.transition_model, self.reward_model)
        
        return
