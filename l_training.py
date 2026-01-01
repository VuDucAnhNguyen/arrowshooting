# train.py
import torch
from arrow_env import ArrowEnv
from observation_wrapper import ObsWrapper
from reward_wrapper import RewardWrapper
from agent import Agent

if __name__ == "__main__":
    base_env = ArrowEnv()
    env = ObsWrapper(RewardWrapper(base_env))
    agent = Agent(env)

    for ep in range(10):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            done = terminated or truncated
        print(f"Episode {ep+1} | Total Reward: {total_reward:.1f} | Score: {info['score']}")
