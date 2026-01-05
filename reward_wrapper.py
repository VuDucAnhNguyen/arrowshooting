import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    
    def __init__ (self, env):
        super().__init__(env)
        self.current_zone = 3 

    def reset(self, **kwargs):
        self.current_zone = 3
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, terminated, truncated, info = self.env.step(action)
        reward = self.calculate_reward(obs, info)
        return obs, reward, terminated, truncated, info
    
    def calculate_reward(self, obs, info):
        reward = 0.0
        step_info = info.get('step_info', {})
        
        # 1. HARD EVENTS
        if step_info.get('shot_fired', False):
            reward -= 0.0 
            
        if step_info.get('hit_priority_target', False):
            reward += 100.0
            self.current_zone = 3
            
        # phạt trượt
        if step_info.get('arrows_went_out', 0) > 0:
            reward -= 5.0 * step_info['arrows_went_out']
            self.current_zone = 3
            
        # 2. shaping reward
        active_arrows = obs.get('arrows', [])
        active_targets = obs.get('targets', [])

        if not active_arrows:
            self.current_zone = 3

        if active_arrows and active_targets:
            priority_target = active_targets[0] 
            t_pos = np.array([priority_target['pos']['x'], priority_target['pos']['y']])
            
            min_dist = float('inf')
            for arrow in active_arrows:
                a_pos = np.array([arrow['pos']['x'], arrow['pos']['y']])
                dist = np.linalg.norm(a_pos - t_pos)
                if dist < min_dist:
                    min_dist = dist
            
            
            # Mốc 1 (< 150px):
            if min_dist < 150 and self.current_zone > 2:
                reward += 1.0
                self.current_zone = 2

            # Mốc 2 (< 100px):
            if min_dist < 100 and self.current_zone > 1:
                reward += 1.0
                self.current_zone = 1

            # Mốc 3 (< 50px):
            if min_dist < 50 and self.current_zone > 0:
                reward += 2.0
                self.current_zone = 0

        # 3. phạt không làm gì
        if len(active_arrows) == 0 and not step_info.get('shot_fired', False):
            reward -= 0.1
            
        # 4. phạt giữ đạn
        arrows_left = step_info.get('arrows_left', 0)
        if arrows_left > 0:
            reward -= 0.01 * arrows_left

        return reward