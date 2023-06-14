import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import random
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HallucinatedModel(nn.Module):
    def __init__(self, base_model, beta=1.0):
        super(HallucinatedModel, self).__init__()
        self.base_model = base_model
        self.beta = beta

        self.original_action_dim = self.base_model.action_dim
        self.action_dim = self.original_action_dim + self.base_model.state_dim

    def forward(self, state, action):
        control_action = action[:, :self.original_action_dim]
        optimism_vars = action[:, self.original_action_dim:]
        optimism_vars = torch.clamp(optimism_vars, -1.0, 1.0)

        mean, stddev = self.base_model.forward(state, control_action)

        return mean + self.beta * (optimism_vars * torch.sqrt(stddev))
    
    def forward_seprate(self, state, action):
        control_action = action[:, :self.original_action_dim]
        optimism_vars = action[:, self.original_action_dim:]
        optimism_vars = torch.clamp(optimism_vars, -1.0, 1.0)

        mean, stddev = self.base_model.forward_seprate(state, control_action)
        mean = torch.stack(mean)
        stddev = torch.stack(stddev)

        return mean + self.beta * (optimism_vars * torch.sqrt(stddev))
    
    def get_next_states(self, state, action):
        return self.forward(state, action)
    
    def get_next_state(self, state, action):
        return self.forward_single(state, action)
    
    def get_next_state_separate(self, state, action):
        return self.forward_seprate(state, action)
    
    def forward_single(self, state, action):
        control_action = action[:self.original_action_dim]
        optimism_var = torch.tensor(action[self.original_action_dim:])
        optimism_var = torch.clamp(optimism_var, -1.0, 1.0)

        mean, stddev = self.base_model.forward(state, control_action)

        return mean + self.beta * (optimism_var * torch.sqrt(stddev))
    
    def train_model(self, T, epochs=1500, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if len(T) > 30:
            # use last trajectory and 29 random ones
            T = [T[-1]] + random.sample(T[:-1], 29)
        for epoch in range(epochs):

            total_loss = 0.0

            for tau in T:
                states, actions = torch.tensor(tau[::2], dtype=torch.float32).to(device), torch.tensor(tau[1::2], dtype=torch.float32).to(device)
                next_states = self.get_next_states(states[:-1], actions[:-1])
                loss = F.mse_loss(next_states, states[1:])
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss /= len(T)

            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")

            # wandb.log({"Transition Model Loss": total_loss})