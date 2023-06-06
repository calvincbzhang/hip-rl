import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianLinear
import logging

class BNNTransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(BNNTransitionModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.linear1 = BayesianLinear(state_dim + action_dim, hidden_dim)
        self.linear2 = BayesianLinear(hidden_dim, hidden_dim)
        self.linear3 = BayesianLinear(hidden_dim, state_dim)

    def forward(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
    
    def train_model(self, T, epochs=500, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if len(T) > 5:
            # choose last 5 trajectories
            T = T[-5:]
        for epoch in range(epochs):

            trajectory_loss = 0.0

            for tau in T:
                states, actions = torch.tensor(tau[::2], dtype=torch.float32), torch.tensor(tau[1::2], dtype=torch.float32)
                next_states = self.forward(states, actions)
                loss = torch.sum((next_states - states)**2)
                trajectory_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {trajectory_loss}")
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {trajectory_loss}")