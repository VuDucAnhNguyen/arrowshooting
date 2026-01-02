# action_wrapper.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ArrowActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )

    def action(self, action):
        # QUAN TRỌNG: Phải clip hành động từ Agent vào khoảng [-1, 1]
        # Vì dist.sample() có thể ra các giá trị như -1.2, 1.5...
        action = np.clip(action, -1.0, 1.0)
        
        # 1. Map Góc: [-1, 1] -> [0, 90]
        angle = (action[0] + 1) / 2 * 90.0
        
        # 2. Map Lực: [-1, 1] -> [10, 50]
        power = (action[1] + 1) / 2 * (50.0 - 10.0) + 10.0
        
        # 3. Map Bắn: [-1, 1] -> [0, 1]
        shoot = 1
        
        return np.array([angle, power, shoot], dtype=np.float64)