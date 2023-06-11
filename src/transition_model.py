import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(TransitionModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_output = nn.Linear(hidden_dim, state_dim)
        self.stddev_output = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mean = self.mean_output(x)
        mean = torch.clamp(mean, -1000, 1000)
        stddev = F.softplus(self.stddev_output(x))
        stddev = torch.clamp(stddev, 1e-6, 1)

        return mean, stddev


class EnsembleTransitionModel(nn.Module):
    """
    Ensamble of TransitionModel with N models (implements a probabilistic ensemble)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=32, N=5):
        super(EnsembleTransitionModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.N = N

        self.models = nn.ModuleList([TransitionModel(state_dim, action_dim, hidden_dim) for _ in range(N)])

    def forward(self, state, action):
        means, stddevs = self.forward_seprate(state, action)

        mean = torch.mean(torch.stack(means), dim=0)
        stddev = torch.mean(torch.stack(stddevs), dim=0)

        return mean, stddev
    
    def forward_seprate(self, state, action):
        means, stddevs = [], []
        for model in self.models:
            mean, stddev = model.forward(state, action)
            means.append(mean)
            stddevs.append(stddev)

        return means, stddevs
    
    def get_next_state(self, state, action):
        mean, stddev = self.forward(state, action)
        next_state = torch.randn_like(mean) * stddev + mean
        return next_state
    
    def get_next_state_seprate(self, state, action):
        means, stddevs = self.forward_seprate(state, action)
        next_states = []
        for mean, stddev in zip(means, stddevs):
            next_state = torch.randn_like(mean) * stddev + mean
            next_states.append(next_state)
        return next_states
    
    def train_model(self, T, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if len(T) > 10:
            # use last trajectory and 9 random ones
            T = [T[-1]] + random.sample(T[:-1], 9)
        for epoch in range(epochs):

            total_loss = 0.0

            for tau in T:
                states, actions = torch.tensor(tau[::2], dtype=torch.float32).to(device), torch.tensor(tau[1::2], dtype=torch.float32).to(device)
                next_states = self.get_next_state(states[:-1], actions[:-1])
                loss = F.mse_loss(next_states, states[1:])
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss /= len(T)

            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")