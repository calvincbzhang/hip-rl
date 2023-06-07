import numpy as np
import torch
import logging

from policy import Policy
from reward_model import BNNRewardModel
from transition_model import BNNTransitionModel
from hallucinated_model import HallucinatedModel
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: Planning function
# TODO: Policy search function

class HIPRL:
    def __init__(self, env, config):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.episodes = config['episodes']
        self.steps = config['steps']

        self.policy = Policy(self.state_dim, self.action_dim).to(device)
        self.reward_model = BNNRewardModel(self.state_dim, self.action_dim).to(device)
        self.base_model = BNNTransitionModel(self.state_dim, self.action_dim).to(device)
        self.hallucinated_model = HallucinatedModel(self.base_model).to(device)

        # trajectories, preferences and rewards
        self.T = []
        self.P = []
        self.R = []
    
    def train(self):

        # execute policy ones to get initial trajectory
        trajectory, cum_reward = self.execute_policy()

        # append
        self.T.append(trajectory)
        self.R.append(cum_reward)

        # train models
        for episode in range(self.episodes):

            print(f"======== Episode {episode+1}/{self.episodes} ========")
            logging.info(f"======== Episode {episode+1}/{self.episodes} ========")
            
            # if episode >= 10:
            #     #  train policy
            #     self.train_policy()

            # execute policy
            trajectory, cum_reward = self.execute_policy()

            # sample trajectory from T and its corresponding reward
            index = np.random.randint(len(self.T))
            trajectory_old = self.T[index]
            cum_reward_old = self.R[index]

            # append
            self.T.append(trajectory)
            self.R.append(cum_reward)

            # compute true preference
            preference = (cum_reward - cum_reward_old) #/ (self.steps)

            # compute episode preference error
            predicted_preference = self.reward_model.get_preference(trajectory, trajectory_old)
            preference_deviation = np.absolute(preference - predicted_preference)
            wandb.log({"preference_deviation": preference_deviation})

            # append preference [trajectory, trajectory_old, preference]
            self.P.append([trajectory, trajectory_old, preference])

            # train models
            self.train_models()

            # if episode >= 10:
            #     # test policy so far
            #     self.test_policy()

    def execute_policy(self):
            
        # initialize trajectory
        trajectory = []

        # reset environment
        state, _ = self.env.reset()
        self.initial_state = state

        # cumulative reward
        cum_reward = 0

        # cumulative transition deviation
        cum_transition_deviation = 0

        # execute policy
        for step in range(self.steps):

            # get action from policy
            mean, var = self.policy(torch.FloatTensor(state).to(device))

            # Sample action from normal distribution
            action = torch.normal(mean, var).detach().numpy()

            # execute action
            next_state, reward, _, _, _ = self.env.step(action)

            # compute step transition error
            predicted_next_state = self.base_model.forward(state, action).detach().numpy()
            transition_deviation = np.sum(np.absolute(next_state - predicted_next_state)) / self.state_dim
            cum_transition_deviation += transition_deviation

            # append to trajectory
            trajectory.append(state)
            trajectory.append(action)

            # update state
            state = next_state

            # update cumulative reward
            cum_reward += reward

        wandb.log({"avg_transition_deviation": cum_transition_deviation / self.steps})

        return trajectory, cum_reward
    
    def train_models(self):
        
        # train reward model
        print("Training reward model...")
        logging.info("Training reward model...")
        self.reward_model.train_model(self.P)
    
        # train transition model
        print("Training transition model...")
        logging.info("Training transition model...")
        self.base_model.train_model(self.T)

    # def train_policy(self):
            
    #     # train policy
    #     print("Training policy...")
    #     logging.info("Training policy...")
    #     self.policy.train_policy(self.initial_state, self.hallucinated_model, self.reward_model)

    # def test_policy(self):
                
    #     # test policy
    #     print("Testing policy...")
    #     logging.info("Testing policy...")
    #     trajectory, cum_reward = self.execute_policy()
    #     print(f"Cumulative reward: {cum_reward}")