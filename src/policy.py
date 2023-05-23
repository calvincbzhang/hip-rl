import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    def train_policy(self, transition_model, reward_model, epochs=5, batch_size=8, learning_rate=1e-3, gamma=0.99):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Simulate trajectories using the transition model and the current policy
        trajectories = self.simulate_model(transition_model, n_trajectories=batch_size)

        for _ in range(epochs):
            optimizer.zero_grad()

            # Compute the loss
            total_loss = 0.0
            for states, actions in trajectories:
                states = torch.stack(states[:-1])
                actions = torch.stack(actions)

                # Compute the reward
                rewards = reward_model.predict(states, actions)

                # Compute actor and critic outputs
                actor_outputs, critic_outputs = self.forward(states)

                # Optimize the critic
                critic_loss = F.mse_loss(critic_outputs, rewards)

                # Optimize the actor
                actor_loss = -torch.mean(critic_outputs.detach() * actor_outputs)

                # print critic_loss and actor_loss
                print("Critic loss: {}".format(critic_loss.item()))
                print("Actor loss: {}".format(actor_loss.item()))

                # Update the total loss
                total_loss += actor_loss + critic_loss

            # Update the parameters of the policy
            total_loss.backward()
            optimizer.step()

            print("Total loss: {}".format(total_loss.item()))

        # Update the parameters of the actor target network
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Update the parameters of the critic target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        return total_loss.item()