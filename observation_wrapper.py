import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        base_env = env.unwrapped

        self.WORLD_WIDTH = getattr(base_env, 'WORLD_WIDTH', 1800.0)
        self.WORLD_HEIGHT = getattr(base_env, 'WORLD_HEIGHT', 900.0)
        self.MAX_MANA = getattr(base_env, 'MAX_MANA', 100.0)
        self.MAX_ARROWS = getattr(base_env, 'MAX_ARROWS', 20.0)
        self.MAX_WIND = getattr(base_env, 'WIND_MAX_STRENGTH', 0.2)

        self.MAX_TARGETS = 5
        self.MAX_ACTIVE_ARROWS = 7 #số mũi tên active tối đa

        input_dim = 6 + (self.MAX_TARGETS * 5) + (self.MAX_ACTIVE_ARROWS * 5)
        # 6: tọa độ agent (x,y), vector gió (x,y), mana, arrows_left
        # Targets: x, y, vx, vy, active
        # Arrows: x, y, vx, vy, active

        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (input_dim, ),
            dtype = np.float32  
        )
    
    def observation(self, obs):
        state = []

        state.append(obs['player']['x'] / self.WORLD_WIDTH)
        state.append(obs['player']['y'] / self.WORLD_HEIGHT)

        state.append(obs['wind']['x'] / self.MAX_WIND)
        state.append(obs['wind']['y'] / self.MAX_WIND)

        state.append(obs['resources']['mana'] / self.MAX_MANA)
        state.append(obs['resources']['arrows_left'] / self.MAX_ARROWS)

        targets = obs['targets']

        for i in range(self.MAX_TARGETS):
            if i < len(targets):
                state.append(targets[i]['pos']['x'] / self.WORLD_WIDTH)
                state.append(targets[i]['pos']['y'] / self.WORLD_HEIGHT)
                state.append(targets[i]['vel']['x'] / 2.0)
                state.append(targets[i]['vel']['y'] / 2.0)
                state.append(1)
            else:
                state.extend([0, 0, 0, 0, 0])

        arrows = obs['arrows']

        sorted_arrows = list(reversed(arrows))
        
        for i in range(self.MAX_ACTIVE_ARROWS):
            if i < len(sorted_arrows):
                state.append(sorted_arrows[i]['pos']['x'] / self.WORLD_WIDTH)
                state.append(sorted_arrows[i]['pos']['y'] / self.WORLD_HEIGHT)
                state.append(sorted_arrows[i]['vel']['x'] / 60.0)
                state.append(sorted_arrows[i]['vel']['y'] / 60.0)
                state.append(1)
            else:
                state.extend([0, 0, 0, 0, 0])

        return np.array(state, dtype=np.float32)

