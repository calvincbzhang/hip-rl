import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import logging

eps = np.finfo(np.float32).eps.item()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Policy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_output = nn.Linear(hidden_dim, action_dim)
        self.var_output = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        mean = self.mean_output(x)
        var = torch.exp(self.var_output(x))  # Ensure variance is positive
        
        return mean, var
    
    def train_policy(self, initial_state, transition_model, reward_model, epochs=1000, steps=100, lr=0.001, gamma=0.99):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            state = initial_state
            rewards = []
            log_probs = []

            for step in range(steps):
                # Forward pass through the policy network
                state = torch.tensor(state, dtype=torch.float32)
                mean, var = self.forward(state)
                dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(var))

                # Sample an action from the distribution
                action = dist.sample()

                # Compute log probability of the action
                log_prob = dist.log_prob(action)

                # Perform the action and observe the next state and reward
                next_state = transition_model(state, action)
                reward = reward_model(state, action)

                rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state
            
            # Compute the discounted rewards
            R = 0
            policy_loss = []
            rtg = []

            for r in rewards[::-1]:
                R = r + gamma * R
                rtg.insert(0, R)

            # Normalize the rewards
            rtg = torch.tensor(rtg)
            rtg = (rtg - rtg.mean()) / (rtg.std() + eps)

            # Compute the policy loss
            for log_prob, reward in zip(log_probs, rtg):
                policy_loss.append(-log_prob * reward)
            
            policy_loss = torch.stack(policy_loss).sum(dim=0).requires_grad_(True)

            # Update the policy network
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Policy Loss: {policy_loss}")
                logging.info(f"Epoch {epoch}/{epochs}, Policy Loss: {policy_loss}")