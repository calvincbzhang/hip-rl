import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPCPolicy(nn.Module):
    def __init__(self, dynamics_model, reward_fn, planning_horizon=20, optim_steps=5, num_candidates=100, top_k=10):
        super(MPCPolicy, self).__init__()
        self.dynamics_model = dynamics_model
        self.reward_fn = reward_fn
        self.planning_horizon = planning_horizon
        self.optim_steps = optim_steps
        self.num_candidates = num_candidates
        self.top_k = top_k

        # regularize covariance
        self.cov_reg = torch.eye(self.planning_horizon * self.dynamics_model.action_dim) * 1e-4

    def reset(self):
        # reset mean to 0 and cov to I
        self.mean = torch.zeros(self.planning_horizon * self.dynamics_model.action_dim)
        self.cov = torch.eye(self.planning_horizon * self.dynamics_model.action_dim)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.reset()

        for i in range(self.optim_steps):
            # sample actions from the current mean and covariance
            top_samples = self.sample_candidates(state)
            # reshape the samples to (top_k, planning_horizon * action_dim)
            top_samples = top_samples.reshape(self.top_k, self.planning_horizon * self.dynamics_model.action_dim)

            self.mean = top_samples.mean(dim=0)
            # compute covariance matrix, shape (planning_horizon * action_dim, planning_horizon * action_dim)
            self.cov = top_samples.T.cov()
            if torch.linalg.matrix_rank(self.cov) < self.cov.shape[0]:
                self.cov += self.cov_reg

        # sample actions from the current mean and covariance
        top_samples = self.sample_candidates(state)

        return top_samples[0, 0]

    def sample_candidates(self, state):
        # sample actions from the current mean and covariance
        samples = torch.distributions.MultivariateNormal(self.mean, self.cov).sample((self.num_candidates,))
        # reshape the samples to (num_candidates, planning_horizon, action_dim)
        samples = samples.reshape(self.num_candidates, self.planning_horizon, self.dynamics_model.action_dim)
        # clamp the actions to -1 and 1
        samples = torch.clamp(samples, -1, 1)

        # evaluate the reward for each action sequence
        rewards = self.evaluate_candidates(state, samples)

        # select the top k actions
        top_indices = rewards.argsort(descending=True)[:self.top_k]

        return samples[top_indices]
    
    def evaluate_candidates(self, state, samples):
        # compute the reward for each action sequence using the dynamics model and reward function
        rewards = torch.zeros(self.num_candidates)

        for i in range(self.num_candidates):
            rewards[i] = self.evaluate_candidate(state, samples[i])

        return rewards
    
    def evaluate_candidate(self, state, actions):
        # compute the reward for the given action sequence using the dynamics model and reward function
        reward = 0

        for i in range(self.planning_horizon):
            mean_state, stddev_state = self.dynamics_model(state, actions[i])
            next_state = torch.randn_like(mean_state) * stddev_state + mean_state

            mean_reward, stddev_reward = self.reward_fn(state, actions[i])
            reward += torch.randn_like(mean_reward) * stddev_reward + mean_reward

            state = next_state

        return reward