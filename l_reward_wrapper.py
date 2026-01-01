import numpy as np
import gymnasium as gym

class ArrowRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, terminated, truncated, info = self.env.step(action)

        reward = 0.0
        step_info = info['step_info']

        # Hit / miss reward
        reward += step_info['targets_hit'] * 50
        reward -= step_info['arrows_went_out'] * 1
        if step_info['shot_fired']:
            reward -= 0.05
        if terminated:
            reward += 100
        if truncated and not terminated:
            reward -= 20

        # ----- New: distance bonus -----
        closest_dist_before = step_info.get('closest_dist_before', None)
        closest_dist_after = step_info.get('closest_dist_after', None)

        # If info gives nothing, compute manually:
        arrow_positions = [np.array([a['pos']['x'], a['pos']['y']]) for a in obs['arrows']]
        target_positions = [np.array([t['pos']['x'], t['pos']['y']]) for t in obs['targets']]
        if arrow_positions and target_positions:
            dists = [np.linalg.norm(a-t) for a in arrow_positions for t in target_positions]
            if dists:
                reward += 0.1 * (step_info.get('closest_dist_before', max(dists)) - min(dists))  # reward getting closer

        return obs, reward, terminated, truncated, info
