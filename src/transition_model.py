import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianLinear
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BNNTransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32, samples=10):
        super(BNNTransitionModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.samples = samples

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
    
    def get_mean_stddev(self, state, action):
        preds = [self.forward(state, action) for _ in range(self.samples)]
        preds = torch.stack(preds)
        mean = torch.mean(preds, dim=0)
        stddev = torch.std(preds, dim=0)
        return mean, stddev
    
    def train_model(self, T, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if len(T) > 10:
            # use last trajectory and 9 random ones
            T = [T[-1]] + random.sample(T[:-1], 9)
        for epoch in range(epochs):

            total_loss = 0.0

            for tau in T:
                states, actions = torch.tensor(tau[::2], dtype=torch.float32).to(device).to(device), torch.tensor(tau[1::2], dtype=torch.float32).to(device).to(device)
                next_states = self.forward(states, actions)
                loss = torch.sum((next_states - states)**2)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss /= len(T)

            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")