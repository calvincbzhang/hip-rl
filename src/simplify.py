import gym
import torch
import gpytorch
import random
import numpy as np

import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


def exact_mll(predicted_distribution, target, gp):
    """Calculate negative marginal log-likelihood of exact model."""
    data_size = target.shape[-1]
    loss = -predicted_distribution.log_prob(target).sum()

    if isinstance(gp, nn.ModuleList):
        for gp_ in gp:
            # Add additional terms (SGPR / learned inducing points,
            # heteroskedastic likelihood models)
            for added_loss_term in gp_.added_loss_terms():
                loss += added_loss_term.loss()

            # Add log probs of priors on the (functions of) parameters
            for _, prior, closure, _ in gp_.named_priors():
                loss += prior.log_prob(closure()).sum()
    else:
        for added_loss_term in gp.added_loss_terms():
            loss += added_loss_term.loss()

        # Add log probs of priors on the (functions of) parameters
        for _, prior, closure, _ in gp.named_priors():
            loss += prior.log_prob(closure()).sum()

    return loss / data_size


class Delta(gpytorch.distributions.Delta):
    """Delta Distribution."""

    def __init__(self, validate_args=False, *args, **kwargs):
        super().__init__(validate_args=validate_args, *args, **kwargs)

    def __str__(self):
        """Get string of Delta distribution."""
        return f"Delta loc: {self.v}"

    def entropy(self):
        """Return entropy of distribution."""
        return torch.zeros(self.batch_shape)
    

def tensor_to_distribution(args, **kwargs):

    if not isinstance(args, tuple):
        return Categorical(logits=args)
    elif torch.all(args[1] == 0):
        if kwargs.get("add_noise", False):
            noise_clip = kwargs.get("noise_clip", np.inf)
            policy_noise = kwargs.get("policy_noise", 1)
            try:
                policy_noise = policy_noise()
            except TypeError:
                pass
            mean = args[0] + (torch.randn_like(args[0]) * policy_noise).clamp(
                -noise_clip, noise_clip
            )
        else:
            mean = args[0]
        return Delta(v=mean, event_dim=min(1, mean.dim()))
    else:
        if kwargs.get("tanh", False):
            d = TransformedDistribution(
                MultivariateNormal(args[0], scale_tril=args[1]), [TanhTransform()]
            )
        else:
            d = MultivariateNormal(args[0], scale_tril=args[1])
        return d


def add_data_to_gp(gp_model, new_inputs, new_targets):
    """Add new data points to an existing GP model.
    Once available, gp_model.get_fantasy_model should be preferred over this.
    """
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    targets = torch.cat((gp_model.train_targets, new_targets), dim=-1)
    gp_model.set_train_data(inputs, targets, strict=False)


def add_input_to_gp(gp_model, new_inputs):
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    gp_model.set_train_data(inputs, strict=False)


class ExactGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, mean=None, kernel=None):
        super().__init__(train_x, train_y, likelihood)
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
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(nn.Module):

    def __init__(self, state, action, target, mean=None, kernel=None):
        super().__init__()
        self.state = state
        self.action = action
        self.target = target
        self.mean = mean
        self.kernel = kernel

        self.state_dim = (state.shape[-1],)
        self.action_dim = (action.shape[-1],)

        train_x, train_y = self.state_actions_to_train_data(state, action, target)

        likelihoods = []
        gps = []
        for train_y_i in train_y:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp = ExactGP(train_x, train_y_i, likelihood, mean, kernel)
            gps.append(gp)
            likelihoods.append(likelihood)

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps)


    def forward(self, state, action):
        """Get next state distribution."""
        test_x = self.state_actions_to_input_data(state, action)

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
        new_x, new_y = self.state_actions_to_train_data(state, action, target)
        for i, new_y_i in enumerate(new_y):
            add_data_to_gp(self.gp[i], new_x, new_y_i)

        
    def _transform_weight_function(self, weight_function=None):
        if not weight_function:
            return None

        def _wf(x):
            s = x[:, : -self.action_dim]
            a = x[:, -self.action_dim :]
            return weight_function(s, a)

        return _wf


    def state_actions_to_input_data(self, state, action):
        # Reshape the training inputs to fit gpytorch-batch mode
        train_x = torch.cat((state, action), dim=-1)
        if train_x.dim() < 2:
            train_x = train_x.unsqueeze(0)

        return train_x
    

    def state_actions_to_train_data(self, state, action, target):
        train_x = self.state_actions_to_input_data(state, action)

        train_y = target.t().contiguous()

        return train_x, train_y
    

class GPReward(nn.Module):

    def __init__(self, state, action, mean=None, kernel=None):
        super().__init__()
        self.state = state
        self.action = action
        self.reard = torch.tensor([1])
        self.mean = mean
        self.kernel = kernel

        self.state_dim = (state.shape[-1],)
        self.action_dim = (action.shape[-1],)
        self.reward_dim = (1,)

        train_x, train_y = self.state_actions_to_train_data(state, action, torch.tensor([1]))

        likelihoods = []
        gps = []
        for train_y_i in train_y:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp = ExactGP(train_x, train_y_i, likelihood, mean, kernel)
            gps.append(gp)
            likelihoods.append(likelihood)

        self.likelihood = torch.nn.ModuleList(likelihoods)
        self.gp = torch.nn.ModuleList(gps)


    def forward(self, state, action):
        """Get next state distribution."""
        test_x = self.state_actions_to_input_data(state, action)

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
        new_x, new_y = self.state_actions_to_train_data(state, action, target)
        for i, new_y_i in enumerate(new_y):
            add_data_to_gp(self.gp[i], new_x, new_y_i)

    
    def add_input(self, state, action):
        new_x = self.state_actions_to_input_data(state, action)
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


    def state_actions_to_input_data(self, state, action):
        # Reshape the training inputs to fit gpytorch-batch mode
        train_x = torch.cat((state, action), dim=-1)
        if train_x.dim() < 2:
            train_x = train_x.unsqueeze(0)

        return train_x


    def state_actions_to_train_data(self, state, action, target):
        train_x = self.state_actions_to_input_data(state, action)

        train_y = target.t().contiguous()

        return train_x, train_y
    

if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v4')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    state = torch.zeros(1, state_dim[0])
    action = torch.zeros(1, action_dim[0])
    next_state = torch.zeros(1, state_dim[0])

    model = ExactGPModel(state, action, next_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    reward_model = GPReward(state, action)
    reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.1)

    preferences = {}
    trajectories = []
    rewards = []

    # TODO: include done condition

    # generate one trajectory
    done = False
    t = []
    episode_reward = 0
    obs = env.reset()[0]

    for time in range(100):
        action = env.action_space.sample()
        next_obs, reward, done, _, info = env.step(action)
        episode_reward += reward

        model.add_data(torch.tensor(obs).unsqueeze(0),
                       torch.tensor(action).unsqueeze(0),
                       torch.tensor(next_obs).unsqueeze(0))
        
        reward_model.add_input(torch.tensor(obs).unsqueeze(0),
                        torch.tensor(action).unsqueeze(0))

        t.extend([action, next_obs])

        obs = next_obs

        if done:
            break

    print("Episode: 0, Reward: {}".format(episode_reward))

    trajectories.append(t)
    rewards.append(episode_reward)
    preferences[str(len(trajectories)-1)] = []

    model.train()
    reward_model.train()

    for ep in range(20):

        t = []
        episode_reward = 0
        obs = env.reset()[0]

        for time in range(100):
            action = env.action_space.sample()
            next_obs, reward, done, _, info = env.step(action)
            episode_reward += reward

            model.add_data(torch.tensor(obs).unsqueeze(0),
                        torch.tensor(action).unsqueeze(0),
                        torch.tensor(next_obs).unsqueeze(0))
            
            reward_model.add_input(torch.tensor(obs).unsqueeze(0),
                            torch.tensor(action).unsqueeze(0))
            
            t.extend([action, next_obs])

            optimizer.zero_grad()

            output = tensor_to_distribution(
                model(torch.tensor(obs).unsqueeze(0), torch.tensor(action).unsqueeze(0))
            )

            with gpytorch.settings.fast_pred_var():
                val = torch.stack(tuple([gp.train_targets for gp in model.gp]), 0)
                loss = exact_mll(output, val, model.gp)
                if time % 10 == 0:
                    print(loss)

            loss.backward()
            optimizer.step()

            obs = next_obs

            if done:
                break

        index = random.randint(0, len(trajectories)-1)
        reward_old = rewards[index]

        trajectories.append(t)
        rewards.append(episode_reward)

        preferences[str(len(trajectories)-1)] = []

        if episode_reward >= reward_old:
            preferences[str(len(trajectories)-1)].append(index)
        else:
            preferences[str(index)].append(len(trajectories)-1)

        # update reward model from preferences
        # for the last added preference, it is either prefferred over a past traj
        # or a past traj is preferred over it
        # then mu(1) is if the new traj is preferred over the past traj
        # and mu(2) is if the past traj is preferred over the new traj
        # p(t_1 > t_2) is the exponantial thingy as a function of the learned reward
        # compute cross entropy loss
        # optimize reward model

        print("Episode: {}, Reward: {}".format(ep+1, episode_reward))

    print(preferences)