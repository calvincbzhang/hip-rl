import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.classic_control import utils

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

class PendulumEnv(PendulumEnv):
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

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()
        self.state = obs
        return obs, {}