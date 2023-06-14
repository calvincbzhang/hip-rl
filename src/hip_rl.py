import numpy as np
import torch
import logging

from sac import SAC
from reward_model import RewardModel
from transition_model import EnsembleTransitionModel
from hallucinated_model import HallucinatedModel
import wandb

import gymnasium as gym

from stable_baselines3 import PPO

from gymnasium.envs.registration import register

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilon = 0.1

class HIPRL:
    def __init__(self, env, config):

        register(
           id='swimmer-v0',
            entry_point='envs.swimmer:SwimmerEnv',
            max_episode_steps=300,
        )

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.env_name = config['env_name']
        self.episodes = config['episodes']
        self.steps = config['steps']

        self.reward_model = RewardModel(self.state_dim, self.action_dim).to(device)
        self.base_model = EnsembleTransitionModel(self.state_dim, self.action_dim).to(device)

        # self.hallucinated_model = HallucinatedModel(self.base_model).to(device)

        self.learned_env = gym.make("swimmer-v0", dynamics_model=self.base_model, reward_fn=self.reward_model)
        self.model = PPO("MlpPolicy", self.learned_env, verbose=1)

        # trajectories, preferences and rewards
        self.T = []
        self.P = []
        self.R = []
    
    def train(self):

        # execute policy onece to get initial trajectory
        trajectory, cum_reward = self.execute_policy()

        # append
        self.T.append(trajectory)
        self.R.append(cum_reward)

        # train models
        for episode in range(self.episodes):

            print(f"======== Episode {episode+1}/{self.episodes} ========")
            logging.info(f"======== Episode {episode+1}/{self.episodes} ========")
            
            # train policy
            if episode + 1 >= 6:
                self.learned_env.close()
                self.learned_env = gym.make("swimmer-v0", dynamics_model = self.base_model, reward_fn = self.reward_model)
                self.learned_env.set_current_state(self.learned_env.reset()[0])
                self.model = PPO("MlpPolicy", self.learned_env, verbose=1)
                self.model.learn(total_timesteps=25000)

            # execute policy
            trajectory, cum_reward = self.execute_policy()

            # if cum_reward is more then the last cum_reward, save policy model
            # if cum_reward > self.R[-1]:
            #     self.policy.save_model("models/best_policy_" + str(self.env_name) + "_")

            # self.policy.save_model("models/last_policy_" + str(self.env_name) + "_")

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

                # save models
                torch.save(self.reward_model.state_dict(), "models/reward_model_" + str(self.env_name) + ".pt")
                torch.save(self.base_model.state_dict(), "models/base_model_" + str(self.env_name) + ".pt")

    def execute_policy(self):

        print("Executing policy...")
        logging.info("Executing policy...")
            
        # initialize trajectory
        trajectory = []

        # reset environment
        state, _ = self.env.reset()

        # cumulative reward
        cum_reward = 0

        # cumulative transition deviation
        cum_transition_deviation = 0

        # execute policy
        for step in range(self.steps):

            # get action from policy
            action, _ = self.model.predict(state, deterministic=False)

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

    # def train_policy(self):
            
    #     # train policy
    #     print("Training policy...")
    #     logging.info("Training policy...")
    #     self.policy.train(self.env, self.base_model, self.reward_model, self.init_states, self.steps)