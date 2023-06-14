import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

class MountainCar(Continuous_MountainCarEnv):
    def __init__(self, dynamics_model, reward_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn
        self.dynamics_model = dynamics_model

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.dynamics_model.action_dim,), dtype=np.float32)

    def step(self, action):

        next_state = self.dynamics_model.get_next_state(self.state, action).detach().cpu().numpy()
        reward = self.reward_fn.get_reward(self.state, action).detach().cpu().numpy()

        self.state = next_state

        info = {}

        if self.render_mode == "human":
            self.render()

        position = self.state[0]
        velocity = self.state[1]

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        return next_state, reward, terminated, False, info
    
    def set_current_state(self, state):
        self.state = state