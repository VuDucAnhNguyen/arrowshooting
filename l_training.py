import gymnasium as gym
from l_reward_wrapper import ArrowRewardWrapper
from l_observation_wrapper import ArrowObsWrapper
from arrow_env import ArrowEnv
import pygame
import time


from stable_baselines3 import PPO

env = ArrowEnv(render_mode=None)
env = ArrowRewardWrapper(env)
env = ArrowObsWrapper(env)

model = PPO("MultiInputPolicy", env, verbose=1)

model.learn(total_timesteps=100_000)

model.save("ppo_arrow")
model = PPO.load("ppo_arrow", env = env)

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    pygame.display.flip()
    time.sleep(0.02)

    done = terminated or truncated

env.close()