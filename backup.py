import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from arrow_env import ArrowEnv
from l_reward_wrappers import RewardWrapper
from l_obs_wrappers import ObsWrapper

def make_env():
    env = ArrowEnv(render_mode="human")
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    env = gym.wrappers.ClipAction(env)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    return env

class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        raw_act = self.locals['actions'][0]

        real_a = 0 + (raw_act[0] + 1.0) * 0.5 * (90 - 0)
        real_p = 10 + (raw_act[1] + 1.0) * 0.5 * (50 - 10)
        
        is_shooting = "SHOOT!" if raw_act[2] > 0 else "---"

        print(f"Step {self.num_timesteps:05d} | Góc: {real_a:4.1f}° | Lực: {real_p:4.1f} | {is_shooting}")

        self.training_env.render()
        
        raw_env = self.training_env.envs[0].unwrapped
        if raw_env.window is not None:
            pygame.display.flip()
            pygame.event.pump()
        return True

env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, callback=RenderCallback())