import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

class HalfCheetahEnv(HalfCheetahEnv):
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

        return next_state, reward, False, False, info
    
    def set_current_state(self, state):
        self.state = state

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.state = observation
        return observation