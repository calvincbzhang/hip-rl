import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import logging


class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RewardModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the reward function parameterized by a neural network
        self.reward_fn = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        reward = self.reward_fn(x)
        return reward.squeeze(-1)

    def compute_loss(self, preferences_data, device='cpu'):
        loss = 0.0
        n_pairs = 0

        for tau1, tau2, true_pref in preferences_data:
            states1, actions1 = torch.tensor(tau1[::2][:-1], dtype=torch.float32, device=device), torch.tensor(tau1[1::2], dtype=torch.float32, device=device)
            states2, actions2 = torch.tensor(tau2[::2][:-1], dtype=torch.float32, device=device), torch.tensor(tau2[1::2], dtype=torch.float32, device=device)

            r_tau1 = self.forward(states1, actions1)
            r_tau2 = self.forward(states2, actions2)

            pref = torch.exp(r_tau1).sum() / (torch.exp(r_tau1).sum() + torch.exp(r_tau2).sum())
            loss += true_pref * torch.log(pref) + (1 - true_pref) * torch.log(1 - pref)
            n_pairs += 1

        loss = -loss / n_pairs
        return loss
    
    def train_step(self, preferences_data, device='cpu'):
        # Compute the loss
        loss = self.compute_loss(preferences_data, device=device)

        # Update the model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train(self, preferences_data, epochs=100, device='cpu'):
        for epoch in range(epochs):
            loss = self.train_step(preferences_data, device=device)
            if epoch == 99:
                print(f"Epoch {epoch + 1} | Loss {loss:.4f}")
                logging.info(f"Epoch {epoch + 1} | Loss {loss:.4f}")

    def predict(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        rewards = self.reward_fn(inputs)
        return rewards.squeeze()