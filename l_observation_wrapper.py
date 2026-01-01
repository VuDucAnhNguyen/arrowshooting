import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ArrowObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Giả sử MAX_TARGETS là số lượng target tối đa mà environment có
        self.MAX_TARGETS = 5  

        # Define observation space
        self.observation_space = spaces.Dict({
            "player": spaces.Box(low=np.array([0, 0]),
                                 high=np.array([100, 100]),  # ví dụ: giới hạn x,y
                                 dtype=np.float32),
            "wind": spaces.Box(low=np.array([-10, -10]),
                               high=np.array([10, 10]),
                               dtype=np.float32),
            "resources": spaces.Box(low=np.array([0, 0, 0]),   # mana, time_left, arrows_left
                                    high=np.array([100, 100, 50]),
                                    dtype=np.float32),
            "targets": spaces.Box(low=0,
                                  high=100,
                                  shape=(self.MAX_TARGETS, 4),  # mỗi target: x, y, vx, vy
                                  dtype=np.float32)
        })

    def observation(self, obs):
        player = np.array([obs['player']['x'], obs['player']['y']], dtype=np.float32)
        wind = np.array([obs['wind']['x'], obs['wind']['y']], dtype=np.float32)
        resources = np.array([
            obs['resources']['mana'], 
            obs['resources']['time_left'], 
            obs['resources']['arrows_left']
        ], dtype=np.float32)

        # targets
        targets = np.zeros((self.MAX_TARGETS, 4), dtype=np.float32)
        for i, t in enumerate(obs['targets']):
            if i >= self.MAX_TARGETS:
                break
            targets[i] = np.array([t['pos']['x'], t['pos']['y'], t['vel']['x'], t['vel']['y']], dtype=np.float32)

        return {
            "player": player,
            "wind": wind,
            "resources": resources,
            "targets": targets
        }



    
# Observation Space:
#     Dictionary chứa:
#     - player: {x, y}
#     - wind: {x, y}
#     - resources: {mana, time_left, arrows_left}
#     - targets: [{pos: {x, y}, vel: {x, y}}, ...]