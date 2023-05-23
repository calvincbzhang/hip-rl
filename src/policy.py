import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Policy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Define the actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Define the actor target network
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Actor optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        # Define the critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Define the critic target network
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Critic optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Copy the parameters of the actor network to the actor target network
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Copy the parameters of the critic network to the critic target network
        self.critic_target.load_state_dict(self.critic.state_dict())

    def forward(self, state):
        actor_output = self.actor(state)
        critic_output = self.critic(state)
        return actor_output, critic_output

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        actor_output, _ = self.forward(state)
        action = actor_output.squeeze().detach().numpy()
        return action
    
    def simulate_model(self, transition_model, n_trajectories=10, max_steps=100):
        trajectories = []

        for _ in range(n_trajectories):
            states = []
            actions = []

            # get initial state
            state = torch.tensor(transition_model.sample_initial_state(), dtype=torch.float32)
            states.append(state)

            for _ in range(max_steps):
                action = torch.tensor(self.sample_action(state), dtype=torch.float32)
                actions.append(action)

                next_state = torch.tensor(transition_model.sample_next_state(state, action), dtype=torch.float32)
                states.append(next_state)

                state = next_state

            trajectories.append((states, actions))

        return trajectories
    
    def soft_update_target_networks(self, network, target_network, tau=0.01):
        for network_param, target_network_param in zip(network.parameters(), target_network.parameters()):
            target_network_param.data.copy_(tau * network_param.data + (1.0 - tau) * target_network_param.data)

    def train_policy(self, transition_model, reward_model, epochs=5, batch_size=4, learning_rate=1e-3, gamma=0.99):
        criterion = nn.MSELoss()
        optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # Sample trajectories using the transition model
            trajectories = self.simulate_model(transition_model, n_trajectories=batch_size)

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            actor_losses = []
            critic_losses = []

            for states, actions in trajectories:
                states = torch.stack(states[:-1])
                actions = torch.stack(actions)

                # Compute the predicted rewards using the reward model
                predicted_rewards = reward_model.predict(states, actions)

                # Compute the state-value targets using the critic target network
                critic_targets = self.critic_target(states)

                # Compute the advantages
                advantages = predicted_rewards - critic_targets.detach()

                # Update the actor network
                optimizer_actor.zero_grad()
                actor_output, _ = self.forward(states)
                actor_loss = -torch.mean(actor_output * advantages.unsqueeze(-1))
                actor_loss.backward(retain_graph=True)
                optimizer_actor.step()
                actor_losses.append(actor_loss.item())

                # Update the critic network
                optimizer_critic.zero_grad()
                _, critic_output = self.forward(states)
                critic_loss = criterion(critic_output.squeeze(-1), predicted_rewards)
                critic_loss.backward(retain_graph=True)
                optimizer_critic.step()
                critic_losses.append(critic_loss.item())

            # Update the target networks using a soft update
            self.soft_update_target_networks(self.actor, self.actor_target, tau=0.01)
            self.soft_update_target_networks(self.critic, self.critic_target, tau=0.01)

            # Print epoch statistics
            avg_actor_loss = np.mean(actor_losses)
            avg_critic_loss = np.mean(critic_losses)
            print(f"Epoch {epoch + 1}/{epochs} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

