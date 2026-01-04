import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, terminated, truncated, info = self.env.step(action)

        reward = 0.0
        reward -= 0.01

        step_info = info['step_info']

        reward -= step_info["shot_fired"] * 1
        reward += step_info["targets_hit"] * 200
        reward -= step_info["arrows_went_out"] * 1

        if terminated:
            reward += 250
            reward += info["arrows_left"]
        if truncated:
            reward -= 10

        return obs, reward, terminated, truncated, info
