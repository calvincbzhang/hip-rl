import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianLinear
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HallucinatedModel(nn.Module):
    def __init__(self, base_model, beta=1.0):
        super(HallucinatedModel, self).__init__()
        self.base_model = base_model
        self.beta = beta

        self.original_action_dim = self.base_model.action_dim
        self.action_dim = self.original_action_dim + self.base_model.state_dim

    def forward(self, state, action):
        control_action = action[:self.original_action_dim]
        optimism_vars = action[self.original_action_dim:]
        optimism_vars = torch.clamp(optimism_vars, -1.0, 1.0)

        mean, stddev = self.base_model.forward(state, control_action)

        return mean + self.beta * (optimism_vars @ torch.sqrt(stddev))