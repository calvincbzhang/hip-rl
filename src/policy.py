import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Policy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Define the actor network
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)

        # Define the critic network
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Compute actor output (mean of action distribution)
        actor_x = F.relu(self.actor_fc1(state))
        actor_output = torch.tanh(self.actor_fc2(actor_x))

        # Compute critic output (state-value)
        critic_x = F.relu(self.critic_fc1(state))
        critic_output = self.critic_fc2(critic_x)

        return actor_output, critic_output

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        actor_output, _ = self.forward(state)
        action = actor_output.squeeze().detach().numpy()
        return action

    def train_policy(self, transition_model, reward_model, epochs=10, batch_size=1, learning_rate=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(epochs):
            optimizer.zero_grad()

            # Generate a batch of transitions using the transition model
            states, actions, next_states = transition_model.generate_batch(batch_size)

            # Compute the predicted rewards using the reward model
            rewards = reward_model.predict(states, actions)

            # Compute actor and critic outputs
            actor_outputs, critic_outputs = self.forward(states)

            # Compute log probabilities of the selected actions
            dist = torch.distributions.Normal(actor_outputs, torch.ones_like(actor_outputs))
            log_probs = dist.log_prob(actions).sum(dim=1)

            # Compute advantage estimates
            advantages = rewards - critic_outputs.squeeze()

            # Compute actor loss
            actor_loss = -(log_probs * advantages.detach()).mean()

            # Compute critic loss (mean squared error)
            critic_loss = F.mse_loss(critic_outputs.squeeze(), rewards)

            # Compute total loss
            total_loss = actor_loss + critic_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()
