import numpy as np
import gymnasium as gym

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_targets = env.MAX_TARGETS
        self.max_arrows = env.MAX_ARROWS
        obs_dim = 2 + 2 + 3 + self.max_targets*4 + self.max_arrows*4
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def observation(self, obs):
        vec = []
        # player
        vec.append(obs['player']['x'])
        vec.append(obs['player']['y'])
        # wind
        vec.append(obs['wind']['x'])
        vec.append(obs['wind']['y'])
        # resources
        vec.append(obs['resources']['mana'])
        vec.append(obs['resources']['time_left'])
        vec.append(obs['resources']['arrows_left'])
        # targets
        for i in range(self.max_targets):
            if i < len(obs['targets']):
                t = obs['targets'][i]
                vec.extend([t['pos']['x'], t['pos']['y'], t['vel']['x'], t['vel']['y']])
            else:
                vec.extend([0,0,0,0])
        # arrows
        for i in range(self.max_arrows):
            if i < len(obs['arrows']):
                a = obs['arrows'][i]
                vec.extend([a['pos']['x'], a['pos']['y'], a['vel']['x'], a['vel']['y']])
            else:
                vec.extend([0,0,0,0])
        return np.array(vec, dtype=np.float32)
