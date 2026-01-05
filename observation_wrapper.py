import numpy as np
import gymnasium as gym
from gymnasium import spaces
from params import params

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        base_env = env.unwrapped

        self.WORLD_WIDTH = getattr(base_env, 'WORLD_WIDTH', 1800.0)
        self.WORLD_HEIGHT = getattr(base_env, 'WORLD_HEIGHT', 900.0)
        self.MAX_MANA = getattr(base_env, 'MAX_MANA', 100.0)
        self.MAX_ARROWS = getattr(base_env, 'MAX_ARROWS', 20.0)
        self.MAX_WIND = getattr(base_env, 'WIND_MAX_STRENGTH', 0.2)

        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (params.input_dim, ),
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
        if len(targets) > 0:
            t = targets[0] # Luôn lấy thằng đầu tiên
            state.append(t['pos']['x'] / self.WORLD_WIDTH)
            state.append(t['pos']['y'] / self.WORLD_HEIGHT)
            state.append(t['vel']['x'] / 2.0)
            state.append(t['vel']['y'] / 2.0)
        else:
            state.extend([0, 0, 0, 0])

        
        active_arrows = obs['arrows']
        if len(active_arrows) > 0:
            # Lấy mũi tên đầu tiên (vì chỉ được phép có 1 cái)
            a = active_arrows[0] 
            state.append(a['pos']['x'] / self.WORLD_WIDTH)
            state.append(a['pos']['y'] / self.WORLD_HEIGHT)
            state.append(a['vel']['x'] / 60.0)
            state.append(a['vel']['y'] / 60.0)
            state.append(1) # Active
        else:
            state.extend([0, 0, 0, 0, 0])

        return np.array(state, dtype=np.float32)

