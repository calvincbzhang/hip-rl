import torch
import torch.nn as nn

import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from util import add_data_to_gp, add_input_to_gp


class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, mean=None, kernel=None):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)

        if mean is None:
            mean = gpytorch.means.ZeroMean()
        self.mean_module = mean

        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = kernel

    @property
    def name(self):
        """Get model name."""
        return self.__class__.__name__

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @output_scale.setter
    def output_scale(self, value):
        """Set output scale."""
        self.covar_module.outputscale = value

    @property
    def length_scale(self):
        """Get length scale."""
        return self.covar_module.base_kernel.lengthscale

    @length_scale.setter
    def length_scale(self, value):
        """Set length scale."""
        self.covar_module.base_kernel.lengthscale = value

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPPreference(nn.Module):

    def __init__(self, state, action, horizon, mean=None, kernel=None):
        super().__init__()
        self.state = state
        self.action = action
        self.horizon = horizon
        self.preference = torch.tensor([1])
        self.mean = mean
        self.kernel = kernel

        train_x, train_y = self.state_actions_to_train_data(state, action, horizon, torch.tensor([1]))

        likelihoods = [gpytorch.likelihoods.BernoulliLikelihood()]
        gps = [GPClassificationModel(train_x)]

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps)


    def forward(self, state, action):
        test_x = self.state_actions_to_input_data(state, action, self.horizon)

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
                likelihood(gp(test_x))
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


    def add_data(self, state, action, target):
        """Add new data to GP-Model, independently to each GP."""
        new_x, new_y = self.state_actions_to_train_data(state, action, self.horizon, target)
        for i, new_y_i in enumerate(new_y):
            add_data_to_gp(self.gp[i], new_x, new_y_i)

    
    def add_input(self, state, action):
        new_x = self.state_actions_to_input_data(state, action, self.horizon)
        for i in range(len(self.gp)):
            add_input_to_gp(self.gp[i], new_x)

        
    def _transform_weight_function(self, weight_function=None):
        if not weight_function:
            return None

        def _wf(x):
            s = x[:, : -self.action_dim]
            a = x[:, -self.action_dim :]
            return weight_function(s, a)

        return _wf


    def state_actions_to_input_data(self, state, action, horizon):
        # Reshape the training inputs to fit gpytorch-batch mode
        train_x = torch.cat((state, action), dim=-1)
        if train_x.dim() < 2:
            train_x = train_x.unsqueeze(0)
        return train_x


    def state_actions_to_train_data(self, state, action, horizon, target):
        train_x = self.state_actions_to_input_data(state, action, horizon)
        train_y = target.t().contiguous()
        return train_x, train_y