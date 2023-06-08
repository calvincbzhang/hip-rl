import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import logging

eps = np.finfo(np.float32).eps.item()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: this is still not right

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Policy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.pi_mean_output = nn.Linear(hidden_dim, action_dim)
        self.pi_stddev_output = nn.Linear(hidden_dim, action_dim)

        self.eta_mean_output = nn.Linear(hidden_dim, state_dim)
        self.eta_stddev_output = nn.Linear(hidden_dim, state_dim)

    def forward(self, state):
        pi_mean, pi_stddev, eta_mean, eta_stddev = self.forward_seprarate(state)

        return self.get_pi_eta(pi_mean, pi_stddev, eta_mean, eta_stddev)
    
    def forward_seprarate(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        pi_mean = self.pi_mean_output(x)
        pi_stddev = F.softplus(self.pi_stddev_output(x))

        eta_mean = self.eta_mean_output(x)
        eta_stddev = F.softplus(self.eta_stddev_output(x))

        return pi_mean, pi_stddev, eta_mean, eta_stddev
    
    def get_pi_eta(self, pi_mean, pi_stddev, eta_mean, eta_stddev):
        pi = torch.randn_like(pi_mean) * pi_stddev + pi_mean
        eta = torch.randn_like(eta_mean) * eta_stddev + eta_mean
        # clamp eta to [-1, 1]
        eta = torch.clamp(eta, -1.0, 1.0)

        log_prob_pi = - ((pi - pi_mean)**2) / (2 * (pi_stddev**2)) - torch.log(pi_stddev) - torch.log(torch.sqrt(2  * torch.tensor(np.pi)))
        log_prob_eta = - ((eta - eta_mean)**2) / (2 * (eta_stddev**2)) - torch.log(eta_stddev) - torch.log(torch.sqrt(2  * torch.tensor(np.pi)))

        return pi, eta, log_prob_pi, log_prob_eta
    
    def train_policy(self, initial_state, transition_model, reward_model, epochs=1000, steps=100, lr=0.001, gamma=0.99):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            state = initial_state
            rewards = []
            log_probs = []

            for step in range(steps):
                # Forward pass through the policy network
                state = torch.tensor(state, dtype=torch.float32)
                pi_mean, pi_stddev, eta_mean, eta_stddev = self.forward_seprarate(state)
                pi, eta, log_prob_pi, log_prob_eta = self.get_pi_eta(pi_mean, pi_stddev, eta_mean, eta_stddev)
                action = torch.cat((pi, eta), dim=-1)

                log_prob = torch.sum(log_prob_pi) + torch.sum(log_prob_eta)

                # Perform the action and observe the next state and reward
                next_state = transition_model(state, action)
                r_mean, r_stddev = reward_model(state, action)
                reward = torch.randn_like(r_mean) * r_stddev + r_mean

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
            
            policy_loss = torch.stack(policy_loss).sum(dim=0)

            # Update the policy network
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Policy Loss: {policy_loss}")
                logging.info(f"Epoch {epoch}/{epochs}, Policy Loss: {policy_loss}")