import numpy as np
import torch
import logging

from sac import SAC
from reward_model import RewardModel
from transition_model import EnsembleTransitionModel
from hallucinated_model import HallucinatedModel
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilon = 0.1

class HIPRL:
    def __init__(self, env, config):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.episodes = config['episodes']
        self.steps = config['steps']

        self.reward_model = RewardModel(self.state_dim, self.action_dim).to(device)
        self.base_model = EnsembleTransitionModel(self.state_dim, self.action_dim).to(device)

        self.hallucinated_model = HallucinatedModel(self.base_model).to(device)

        self.policy = SAC(self.state_dim, self.action_dim, env.action_space, hallucinated=True)
        self.policy.to(device)

        # trajectories, preferences and rewards
        self.T = []
        self.P = []
        self.R = []
    
    def train(self):

        random = True

        # get a set of initial states of length 100
        self.init_states = [self.env.reset()[0] for _ in range(100)]

        # execute policy ones to get initial trajectory
        trajectory, cum_reward = self.execute_policy(random=random)

        # append
        self.T.append(trajectory)
        self.R.append(cum_reward)

        # train models
        for episode in range(self.episodes):

            print(f"======== Episode {episode+1}/{self.episodes} ========")
            logging.info(f"======== Episode {episode+1}/{self.episodes} ========")
            
            # train policy
            if episode + 1 >= 6:
                random = False
                self.train_policy()

            # execute policy
            trajectory, cum_reward = self.execute_policy(random=random)

            print(f"cum_reward: {cum_reward}")
            logging.info(f"cum_reward: {cum_reward}")
            wandb.log({"cum_reward": cum_reward})

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
            print(f"preference_deviation: {preference_deviation}")
            logging.info(f"preference_deviation: {preference_deviation}")
            wandb.log({"preference_deviation": preference_deviation})

            # append preference [trajectory, trajectory_old, preference]
            self.P.append([trajectory, trajectory_old, preference])

            # train models
            if episode + 1 >= 5:
                self.train_models()

    def execute_policy(self, random=False):

        print("Executing policy...")
        logging.info("Executing policy...")
            
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

            # save image of render
            img = self.env.render()
            wandb.log({"render": wandb.Image(img)})

            # get action from policy
            # epsilon greedy
            if random or np.random.uniform() < epsilon:
                action = self.policy.select_action(torch.FloatTensor(state).to(device))
            else:
                action = self.policy.select_action_deterministic(torch.FloatTensor(state).to(device))
                
            action = action.cpu().detach().numpy()[:self.action_dim]

            next_state, reward, _, _, _ = self.env.step(action)

            # compute step transition error
            predicted_next_state = (self.base_model.get_next_state(torch.FloatTensor(state).to(device), torch.FloatTensor(action).to(device))).cpu().detach().numpy()
            transition_deviation = np.sqrt(np.sum((next_state - predicted_next_state)**2))
            cum_transition_deviation += transition_deviation

            # append to trajectory
            trajectory.append(state)
            trajectory.append(action)

            # update state
            state = next_state

            # update cumulative reward
            cum_reward += reward

        print(f"cum_transition_deviation: {cum_transition_deviation / self.steps}")
        logging.info(f"cum_transition_deviation: {cum_transition_deviation / self.steps}")
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

    def train_policy(self):
            
        # train policy
        print("Training policy...")
        logging.info("Training policy...")
        self.policy.train(self.hallucinated_model, self.reward_model, self.init_states)