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
    
    def get_next_state(self, state, action):
        return self.forward(state, action)
    
    def get_next_state_separate(self, state, action):
        return self.forward_seprate(state, action)
    
    def forward_single(self, state, action):
        control_action = action[:self.original_action_dim]
        optimism_var = action[self.original_action_dim:]
        optimism_var = torch.clamp(optimism_var, -1.0, 1.0)

        mean, stddev = self.base_model.forward(state, control_action)

        return mean + self.beta * (optimism_var * torch.sqrt(stddev))