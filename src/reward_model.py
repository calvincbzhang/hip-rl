import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianLinear
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BNNRewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32, samples=10):
        super(BNNRewardModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.samples = samples

        self.linear1 = BayesianLinear(state_dim + action_dim, hidden_dim)
        self.linear2 = BayesianLinear(hidden_dim, hidden_dim)
        self.linear3 = BayesianLinear(hidden_dim, 1)

    def forward(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.tanh(self.linear3(x))
    
    def compute_loss(self, P):
        loss = 0.0
        n_pairs = 0

        for tau1, tau2, preference in P:
            states1, actions1 = torch.tensor(tau1[::2], dtype=torch.float32).to(device), torch.tensor(tau1[1::2], dtype=torch.float32).to(device)
            states2, actions2 = torch.tensor(tau2[::2], dtype=torch.float32).to(device), torch.tensor(tau2[1::2], dtype=torch.float32).to(device)

            r_tau1 = self.forward(states1, actions1)
            r_tau2 = self.forward(states2, actions2)

            pref = (torch.sum(r_tau1) - torch.sum(r_tau2)) #/ len(r_tau1)
            loss += ((pref - preference)**2)
            n_pairs += 1

        loss = loss / n_pairs
        return loss
    
    def train_model(self, P, epochs=10, lr=0.01):
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

    def get_preference(self, tau1, tau2):
        states1, actions1 = torch.tensor(tau1[::2], dtype=torch.float32).to(device), torch.tensor(tau1[1::2], dtype=torch.float32).to(device)
        states2, actions2 = torch.tensor(tau2[::2], dtype=torch.float32).to(device), torch.tensor(tau2[1::2], dtype=torch.float32).to(device)

        r_tau1 = self.forward(states1, actions1)
        r_tau2 = self.forward(states2, actions2)

        pref = (torch.sum(r_tau1) - torch.sum(r_tau2)) #/ len(r_tau1)
        return pref.item()
