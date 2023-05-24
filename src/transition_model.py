import gpytorch
import torch
from torch import nn
import numpy as np

import logging


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        if mean is None:
            mean = gpytorch.means.ZeroMean()
        self.mean_module = mean

        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = kernel.to(train_x.device)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPTransitionModel(nn.Module):
    def __init__(self, state, action, next_state, mean=None, kernel=None, device='cpu'):
        super().__init__()
        self.state = state.to(device)
        self.action = action.to(device)
        self.next_state = next_state.to(device)
        self.mean = mean
        self.kernel = kernel

        self.state_dim = (state.shape[-1],)
        self.action_dim = (action.shape[-1],)

        train_x, train_y = self.state_actions_to_train_data(state, action, next_state)

        likelihoods = []
        gps = []
        for train_y_i in train_y:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            gp = ExactGPModel(train_x, train_y_i, likelihood, mean, kernel).to(device)
            gps.append(gp)
            likelihoods.append(likelihood)

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps).to(device)


    def forward(self, state, action):
        input = self.state_actions_to_train_data(state, action)

        if self.training:
            out = [
                likelihood(gp(gp.train_inputs[0]))
                for gp, likelihood in zip(self.gp, self.likelihood)
            ]

            mean = torch.stack(tuple(o.mean for o in out), dim=0)
            scale_tril = torch.stack(tuple(o.scale_tril for o in out), dim=0)
            return mean, scale_tril
        else:
            out = [
                likelihood(gp(input))
                for gp, likelihood in zip(self.gp, self.likelihood)
            ]
            mean = torch.stack(tuple(o.mean for o in out), dim=-1)

            # Sometimes, gpytorch returns negative variances due to numerical errors.
            # Hence, clamp the output variance to the noise of the likelihood.
            stddev = torch.stack(
                tuple(
                    torch.sqrt(o.variance.clamp(l.noise.item() ** 2, float("inf")))
                    for o, l in zip(out, self.likelihood)
                ),
                dim=-1,
            )
            return mean, torch.diag_embed(stddev)


    def add_data(self, state, action, next_state):
        x, y = self.state_actions_to_train_data(state, action, next_state)

        for i, y_i in enumerate(y):
            inputs = torch.cat((self.gp[i].train_inputs[0], x), dim=0)
            outputs = torch.cat((self.gp[i].train_targets.to(self.state.device), torch.tensor([y_i]).to(self.state.device)), dim=-1)
            self.gp[i].set_train_data(inputs, outputs, strict=False)


    def train(self, epochs=100, lr=0.1, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = 0

            for gp in self.gp:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
                output = gp(gp.train_inputs[0])
                loss += -mll(output, gp.train_targets).sum()

            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} \t Loss: {loss.item():.4f}")
                logging.info(f"Epoch {epoch + 1}/{epochs} \t Loss: {loss.item():.4f}")


    def state_actions_to_train_data(self, state, action, next_state=None):
        train_x = torch.cat((state, action), dim=-1).to(state.device)
        if train_x.dim() < 2:
            train_x = train_x.unsqueeze(0)
        if next_state is None:
            return train_x
        train_y = next_state.t().contiguous().to(state.device)
        return train_x, train_y
    

    # get a random initial state
    def sample_initial_state(self):
        return torch.rand(self.state_dim).to(self.state.device)
    

    # sample a next state given a state and action
    def sample_next_state(self, state, action):
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad():
            input = self.state_actions_to_train_data(state, action)
            out = [
                likelihood(gp(input))
                for gp, likelihood in zip(self.gp, self.likelihood)
            ]
            mean = torch.stack(tuple(o.mean for o in out), dim=-1)
            stddev = torch.stack(
                tuple(
                    torch.sqrt(o.variance.clamp(l.noise.item() ** 2, float("inf")))
                    for o, l in zip(out, self.likelihood)
                ),
                dim=-1,
            )
        return torch.normal(mean, stddev).squeeze(0)


    def calibrate_model(self):
        # Add calibration code here
        pass
