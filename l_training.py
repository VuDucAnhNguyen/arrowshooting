import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from arrow_env import ArrowEnv
from other_reward_wrappers import RewardWrapper
from l_obs_wrappers import ObsWrapper

def make_env():
    env = ArrowEnv(render_mode="human")
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    env = gym.wrappers.ClipAction(env)
    env = RewardWrapper(env)
    env = ObsWrapper(env)
    env = Monitor(env) #rollout
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

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeLoggerCallback, self).__init__(verbose)
        self.episode_count = 0
        self.last_score = 0
        self.last_hits = 0

    def _on_step(self) -> bool:
        raw_env = self.training_env.envs[0].unwrapped
        done = self.locals['dones'][0]
        if not done:
            self.last_score = getattr(raw_env, 'score', 0)
            self.last_hits = getattr(raw_env, 'targets_hit', 0)
        else:
            self.episode_count += 1
            print(f"Ep {self.episode_count:4d} | Final Score: {self.last_score:4d} | Hits: {self.last_hits}/5")
            self.last_score, self.last_hits = 0, 0
        return True

env = DummyVecEnv([make_env])
path = "training_result"
log_dir = "./ppo_arrow_tensorboard/"

if os.path.exists(path + ".zip"):
    print("Tim thay model cu, dang tai de hoc tiep...")
    model = PPO.load(path, env = env, tensorboard_log=log_dir, verbose=0)
else:
    print("Khong tim thay model, dang tao moi...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=128,   
        ent_coef=0.005,     
        verbose=0,
        tensorboard_log=log_dir
    )

checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # 10000 steps
    save_path="./logs/",
    name_prefix="ppo_arrow"
)

callback_list = CallbackList([EpisodeLoggerCallback(), checkpoint_callback])
callback_list2 = CallbackList([RenderCallback(), checkpoint_callback])


print("Bat dau qua trinh huan luyen...")
# model.learn(
#     total_timesteps=500000, 
#     callback=callback_list,
#     reset_num_timesteps=False
# )
model.learn(total_timesteps=100000, callback=callback_list2)

model.save(path)
print("--- Da luu thanh cong model moi ---")
