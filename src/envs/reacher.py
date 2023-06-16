import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv

class ReacherEnv(ReacherEnv):
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
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        state = self._get_obs()
        self.state = state
        return state