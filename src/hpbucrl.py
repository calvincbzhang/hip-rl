import numpy as np
import torch

from transition_model import GPTransitionModel


class HPbUCRL:
    def __init__(self, env, config):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        # TODO: initialize parameters
        self.horizon = config['horizon']
        self.train_epochs = config['train_epochs']

        state = torch.zeros(1, self.state_dim)
        action = torch.zeros(1, self.n_actions)
        next_state = torch.zeros(1, self.state_dim)

        self.transition_model = GPTransitionModel(state, action, next_state)
        # TODO: initialize reward model
        self.reward_model = None
        # TODO: initialize policy
        self.policy = None

        self.T = []
        self.R = []
        self.P = {}
    
    def train(self):
        # append random initial trajectory to T
        s = self.env.reset()[0]
        tau = [s]
        reward = 0
        for _ in range(self.horizon):
            a = self.env.action_space.sample()
            s_next, r, _, _, _ = self.env.step(a)

            # add data to the transition model
            self.transition_model.add_data(torch.tensor(s), torch.tensor(a), torch.tensor(s_next))

            s = s_next
            reward += r
            tau.append((a, s_next))

        self.T.append(tau)
        self.R.append(reward)
        self.P[0] = []

        # train for n_episodes
        for k in range(self.train_epochs):
            s = self.env.reset()[0]
            tau = [s]
            reward = 0
            for t in range(self.horizon):
                # TODO: sample action from policy and to plan step in algorithm
                a = self.env.action_space.sample()
                s_next, r, _, _, _ = self.env.step(a)

                # add data to the transition model
                self.transition_model.add_data(torch.tensor(s), torch.tensor(a), torch.tensor(s_next))

                s = s_next
                reward += r
                tau.append((a, s_next))

            # sample old trajectory and its reward
            idx = np.random.randint(len(self.T))
            reward_old = self.R[idx]

            self.T.append(tau)
            self.R.append(reward)

            # TODO: add stochasticity to the preference
            # compute binary preference between new and old trajectory
            if reward > reward_old:
                self.P[k+1] = [idx]
            else:
                self.P[k+1] = []
                self.P[idx].append(k+1)

            # TODO: estimate reward
            # estimate transition model
            self.transition_model.train()
            print("===========================================================")
            # TODO: estimate policy
        
        return
