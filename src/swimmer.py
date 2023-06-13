import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SwimmerEnv(gym.envs.mujoco.swimmer_v4.SwimmerEnv):
    def __init__(self, reward_fn, *args, **kwargs):
        super().__init__(reward_fn, *args, **kwargs)
        self.reward_fn = reward_fn

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        # reward = forward_reward - ctrl_cost

        state = observation

        reward = self.reward_fn.get_reward(state, action)

        info = {
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info