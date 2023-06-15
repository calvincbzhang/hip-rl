import numpy as np
import torch
import random
import logging

from sac import SAC
from reward_model import RewardModel
from transition_model import EnsembleTransitionModel
from hallucinated_model import HallucinatedModel
import wandb

import gymnasium as gym

import stable_baselines3 as sb3
from stable_baselines3 import PPO, TD3
from wandb.integration.sb3 import WandbCallback

# Set a fixed seed
seed = 42 

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from gymnasium.envs.registration import register

register(
    id='LearnedAnt-v4',
    entry_point='envs.ant:AntEnv',
)

register(
    id='LearnedSwimmer-v4',
    entry_point='envs.swimmer:SwimmerEnv',
)

register(
    id='LearnedHalfCheetah-v4',
    entry_point='envs.half_cheetah:HalfCheetahEnv',
)

register(
    id='LearnedHopper-v4',
    entry_point='envs.hopper:HopperEnv',
)

register(
    id='LearnedMountainCarContinuous-v0',
    entry_point='envs.mountain_car:MountainCar',
)

register(
    id='LearnedInvertedPendulum-v4',
    entry_point='envs.inverted_pendulum:InvertedPendulum',
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilon = 0.1

class HIPRL:
    def __init__(self, env, config, foldername):

        self.foldername = foldername

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.env_name = config['env_name']
        self.episodes = config['episodes']
        self.steps = config['steps']

        self.config = config

        self.base_model = EnsembleTransitionModel(self.state_dim, self.action_dim).to(device)
        self.hallucinated_model = HallucinatedModel(self.base_model).to(device)
        self.reward_model = RewardModel(self.state_dim, self.action_dim + self.state_dim).to(device)

        self.learned_env = gym.make("Learned" + self.env_name, dynamics_model=self.hallucinated_model, reward_fn=self.reward_model)

        if self.env_name == "HalfCheetah-v4":
            self.model = PPO(
                "MlpPolicy",
                self.learned_env,
                verbose=1,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                clip_range=self.config['clip_range'],
                ent_coef=self.config['ent_coef'],
                vf_coef=self.config['vf_coef'],
                max_grad_norm=self.config['max_grad_norm'],
                policy_kwargs={"log_std_init": -2, "ortho_init": False, "activation_fn": torch.nn.ReLU, "net_arch": [{"pi": [256, 256], "vf": [256, 256]}]},
            )
        elif self.env_name == "Ant-v4":
            self.model = TD3(
                "MlpPolicy",
                self.learned_env,
                verbose=1,
                learning_starts=self.config['learning_starts'],
            )
        elif self.env_name == "MountainCarContinuous-v0":
            self.model = TD3(
                "MlpPolicy",
                self.learned_env,
                verbose=1,
                action_noise=sb3.common.noise.OrnsteinUhlenbeckActionNoise(mean=np.zeros(self.action_dim), sigma=0.5 * np.ones(self.action_dim)),
            )
        elif self.env_name == "Hopper-v4":
            self.model = TD3(
                "MlpPolicy",
                self.learned_env,
                verbose=1,
                learning_rate=self.config['learning_rate'],
                learning_starts=self.config['learning_starts'],
                batch_size=self.config['batch_size'],
                train_freq=self.config['train_freq'],
                gradient_steps=self.config['gradient_steps'],
            )
        else:
            self.model = PPO(
                "MlpPolicy",
                self.learned_env,
                verbose=1,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                clip_range=self.config['clip_range'],
                ent_coef=self.config['ent_coef'],
                vf_coef=self.config['vf_coef'],
                max_grad_norm=self.config['max_grad_norm'],
            )

        # trajectories, preferences and rewards
        self.T = []
        self.P = []
        self.R = []
    
    def train(self):

        # execute policy onece to get initial trajectory
        trajectory, cum_reward = self.execute_policy()

        self.best_reward = cum_reward

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
                self.learned_env = gym.make("Learned" + self.env_name, dynamics_model = self.hallucinated_model, reward_fn = self.reward_model)

                self.model.set_env(self.learned_env)

                self.model.learn(total_timesteps=self.config['total_timesteps'], callback=WandbCallback(model_save_path=self.foldername), progress_bar=True)

                # evaluate policy
                rewards = self.evaluate_policy(eval_episodes=10)

                # compute average and standard deviation of rewards
                avg_reward = np.mean(rewards)
                stddev_reward = np.std(rewards)
                
                print(f"avg_reward: {avg_reward}")
                logging.info(f"avg_reward: {avg_reward}")
                wandb.log({"avg_reward_evaluation": avg_reward}, commit=False)

                print(f"stddev_reward: {stddev_reward}")
                logging.info(f"stddev_reward: {stddev_reward}")
                wandb.log({"stddev_reward_evaluation": stddev_reward}, commit=False)

                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.model.save(f"{self.foldername}/best_model_{str(self.env_name)}")

            # execute policy
            trajectory, cum_reward = self.execute_policy()

            print(f"cum_reward: {cum_reward}")
            logging.info(f"cum_reward: {cum_reward}")
            if episode + 1 >= 6:
                wandb.log({"cum_reward": cum_reward}, commit=False)

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
            if episode + 1 >= 6:
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
                torch.save(self.reward_model.state_dict(), self.foldername + "/reward_model_" + str(self.env_name) + ".pt")
                torch.save(self.hallucinated_model.state_dict(), self.foldername + "/hallucinated_model_" + str(self.env_name) + ".pt")

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

            next_state, reward, terminated, truncated, _ = self.env.step(action[:self.action_dim])

            # compute step transition error
            predicted_next_state = (self.hallucinated_model.get_next_state(torch.FloatTensor(state).to(device), torch.FloatTensor(action).to(device))).cpu().detach().numpy()
            transition_deviation = np.sqrt(np.sum((next_state - predicted_next_state)**2))
            cum_transition_deviation += transition_deviation

            # append to trajectory
            trajectory.append(state)
            trajectory.append(action)

            # update state
            state = next_state

            # update cumulative reward
            cum_reward += reward

            if terminated or truncated:
                break

        print(f"cum_transition_deviation: {cum_transition_deviation / self.steps}")
        logging.info(f"cum_transition_deviation: {cum_transition_deviation / self.steps}")
        wandb.log({"avg_transition_deviation": cum_transition_deviation / self.steps}, commit=False)

        return trajectory, cum_reward
    
    def train_models(self):
        
        # train reward model
        print("Training reward model...")
        logging.info("Training reward model...")
        self.reward_model.train_model(self.P)
    
        # train transition model
        print("Training transition model...")
        logging.info("Training transition model...")
        self.hallucinated_model.train_model(self.T)

    def evaluate_policy(self, eval_episodes=10):
            
            print("Evaluating policy...")
            logging.info("Evaluating policy...")
            
            # initialize array of rewards
            rewards = []
            
            # execute policy
            for episode in range(eval_episodes):
                
                # reset environment
                state, _ = self.env.reset()
                
                # cumulative reward
                cum_reward = 0
                
                # execute policy
                for step in range(self.steps):
                    
                    # get action from policy
                    action, _ = self.model.predict(state, deterministic=True)
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(action[:self.action_dim])
                    
                    # update state
                    state = next_state
                    
                    # update cumulative reward
                    cum_reward += reward

                    if terminated or truncated:
                        break 
                
                # append reward
                rewards.append(cum_reward)

            return rewards

    # def train_policy(self):
            
    #     # train policy
    #     print("Training policy...")
    #     logging.info("Training policy...")
    #     self.policy.train(self.env, self.base_model, self.reward_model, self.init_states, self.steps)