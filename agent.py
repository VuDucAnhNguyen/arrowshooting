# agent.py
import torch
import torch.optim as optim
import numpy as np
from model import PPO
from params import params

class Agent:
    def __init__(self, env):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.model = PPO(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr)

    # chuẩn hóa action từ [-1, 1] -> [low, high]
    def scale_action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high
        scaled = low + (action + 1) * 0.5 * (high - low)
        return np.clip(scaled, low, high)

    # lấy action (chuẩn hóa nó)
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        dist, _ = self.model(state_tensor)
        action = dist.sample().detach().numpy()[0]
        return self.scale_action(action)
