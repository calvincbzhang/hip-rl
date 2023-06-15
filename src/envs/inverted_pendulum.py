import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv

class InvertedPendulum(InvertedPendulumEnv):
    def __init__(self, dynamics_model, reward_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn
        self.dynamics_model = dynamics_model

        self.action_space = spaces.Box(low=-3, high=3, shape=(self.dynamics_model.action_dim,), dtype=np.float32)

    def step(self, action):

        next_state = self.dynamics_model.get_next_state(self.state, action).detach().cpu().numpy()
        reward = self.reward_fn.get_reward(self.state, action).detach().cpu().numpy()

        self.state = next_state

        info = {}

        if self.render_mode == "human":
            self.render()

        terminated = bool(not np.isfinite(self.state).all() or (np.abs(self.state[1]) > 0.2))

        return next_state, reward, terminated, False, info
    
    def set_current_state(self, state):
        self.state = state

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        self.state = obs
        return obs