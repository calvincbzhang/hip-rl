# Code readapted from https://github.com/Xingyu-Lin/mbpo_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=32):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # x1 = torch.tanh(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        # x2 = torch.tanh(x2)

        return x1, x2


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_space, hidden_dim=32):
        super(Policy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_space = action_space

        self.min_low = np.min(self.action_space.low)
        self.max_high = np.max(self.action_space.high)

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_output = nn.Linear(hidden_dim, action_dim)
        self.stddev_output = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        mean = self.mean_output(x)
        mean = torch.clamp(mean, self.min_low, self.max_high)
        stddev = F.softplus(self.stddev_output(x))
        stddev = torch.clamp(stddev, 1e-6, 1)

        return mean, stddev

    def sample(self, state):

        mean, stddev = self.forward(state)
        
        normal = Normal(mean, stddev)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias  # bound the action

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        return action, log_prob


class SAC(object):
    def __init__(self, state_dim, action_dim, action_space, gamma=0.99, tau=0.05, alpha=0.2, hidden_dim=32, lr=0.0003):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = 10

        self.critic = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.policy = Policy(state_dim, action_dim, action_space, hidden_dim)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        action, _ = self.policy.sample(state)
        return action
    
    def train(self, dynamics_model, reward_fn, init_states, horizon=1000, epochs=500, batch_size=256):

        for epoch in range(epochs):

            state = torch.FloatTensor(init_states)
            
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []

            for t in range(horizon):
                    
                action = self.select_action(state)
                next_state = dynamics_model.get_next_state(state, action)
                reward = reward_fn.get_reward(state, action)

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)

                state = next_state

            # sample random indices
            indices = np.random.randint(0, horizon, size=batch_size)

            # convert to tensors
            state_batch = torch.stack(state_batch)[indices]
            action_batch = torch.stack(action_batch)[indices]
            reward_batch = torch.stack(reward_batch)[indices]
            next_state_batch = torch.stack(next_state_batch)[indices]

            qf1_loss, qf2_loss, policy_loss = self.update_parameters((state_batch, action_batch, reward_batch, next_state_batch), epoch)

            if (epoch+1) % 10 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, QF1 Loss: {qf1_loss}, QF2 Loss: {qf2_loss}, Policy Loss: {policy_loss}")
                logging.info(f"Epoch: {epoch+1}/{epochs}, QF1 Loss: {qf1_loss}, QF2 Loss: {qf2_loss}, Policy Loss: {policy_loss}")



    def update_parameters(self, memory, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch = memory

        # reshape tensors
        state_batch = state_batch.reshape(-1, state_batch.shape[-1])
        action_batch = action_batch.reshape(-1, action_batch.shape[-1])
        reward_batch = reward_batch.reshape(-1, reward_batch.shape[-1])
        next_state_batch = next_state_batch.reshape(-1, next_state_batch.shape[-1])

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        if qf1_loss > 1000000 or qf2_loss > 1000000:
            print(f"QF1 Loss: {qf1_loss}, QF2 Loss: {qf2_loss}")
            print(next_q_value)

        pi, log_pi = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = torch.mean(((self.alpha * log_pi) - min_qf_pi)) # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        self.policy_optim.zero_grad()

        (policy_loss + qf1_loss + qf2_loss).backward()

        self.critic_optim.step()
        self.policy_optim.step()


        if updates % self.target_update_interval == 0:
            # Soft update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()