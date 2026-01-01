import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ArrowActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Định nghĩa lại action_space là [-1, 1] để PPO dễ học
        # Shape là (3,) vì có: Góc, Lực, Bắn
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )

    def action(self, action):
        """
        Hàm này tự động chạy mỗi khi bạn gọi env.step(action)
        Nó sẽ biến đổi action từ [-1, 1] sang dải env mong muốn.
        """
        clipped_action = np.clip(action, -1.0, 1.0)

        # 1. Map Góc: [-1, 1] -> [0, 90]
        # Công thức: (x + 1) / 2 * (max - min) + min
        angle = (clipped_action[0] + 1) / 2 * 90.0
        
        # 2. Map Lực: [-1, 1] -> [10, 50]
        power = (clipped_action[1] + 1) / 2 * (50.0 - 10.0) + 10.0
        
        shoot_signal = clipped_action[2] 
        shoot = 1.0 if shoot_signal > 0.0 else 0.0
        
        # Trả về mảng đúng định dạng env gốc yêu cầu
        return np.array([angle, power, shoot], dtype=np.float64)