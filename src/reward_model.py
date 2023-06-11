import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(RewardModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_output = nn.Linear(hidden_dim, 1)
        self.stddev_output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mean = self.mean_output(x)
        mean = torch.clamp(mean, -1000, 1000)
        stddev = torch.exp(self.stddev_output(x))
        stddev = torch.clamp(stddev, 1e-6, 1)

        return mean, stddev
    
    def get_reward(self, state, action):
        mean, stddev = self.forward(state, action)
        reward = torch.randn_like(mean) * stddev + mean
        return reward
    
    def get_preference(self, tau1, tau2):
        states1, actions1 = torch.tensor(tau1[::2], dtype=torch.float32).to(device), torch.tensor(tau1[1::2], dtype=torch.float32).to(device)
        states2, actions2 = torch.tensor(tau2[::2], dtype=torch.float32).to(device), torch.tensor(tau2[1::2], dtype=torch.float32).to(device)

        mean_tau1, stddev_tau1 = self.forward(states1, actions1)
        mean_tau2, stddev_tau2 = self.forward(states2, actions2)

        r_tau1 = torch.randn_like(mean_tau1) * stddev_tau1 + mean_tau1
        r_tau2 = torch.randn_like(mean_tau2) * stddev_tau2 + mean_tau2

        pref = (torch.sum(r_tau1) - torch.sum(r_tau2)) #/ len(r_tau1)
        return pref.item()
    
    def get_preference_tensor(self, tau1, tau2):
        states1, actions1 = tau1[::2], tau1[1::2]
        states2, actions2 = tau2[::2], tau2[1::2]

        mean_tau1, stddev_tau1 = self.forward(states1, actions1)
        mean_tau2, stddev_tau2 = self.forward(states2, actions2)

        r_tau1 = torch.randn_like(mean_tau1) * stddev_tau1 + mean_tau1
        r_tau2 = torch.randn_like(mean_tau2) * stddev_tau2 + mean_tau2

        pref = (torch.sum(r_tau1) - torch.sum(r_tau2)) #/ len(r_tau1)
        return pref
    
    def compute_loss(self, P):
        loss = 0.0
        n_pairs = 0

        for tau1, tau2, preference in P:
            pref = self.get_preference_tensor(tau1, tau2)
            loss += ((pref - preference)**2)
            n_pairs += 1

        loss = loss / n_pairs
        return loss
    
    def train_model(self, P, epochs=2, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if len(P) > 100:
            # use last preference and sample 99 random preferences
            P = [P[-1]] + random.sample(P[:-1], 99)

        for epoch in range(epochs * len(P)):

                loss = self.compute_loss(P)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs * len(P)}, Loss: {loss.item()}")
                    logging.info(f"Epoch {epoch+1}/{epochs * len(P)}, Loss: {loss.item()}")