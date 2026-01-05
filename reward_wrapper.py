import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    
    def __init__ (self, env):
        super().__init__(env)

    def step(self, action):
        obs, terminated, truncated, info = self.env.step(action)
        reward = self.calculate_reward(obs, info)
        return obs, reward, terminated, truncated, info
    
    def calculate_reward(self, obs, info):
        reward = 0.0
        step_info = info.get('step_info', {})
        
        # 1. HARD EVENTS
        # Miễn phí khai hỏa (0.0) để dụ bắn
        if step_info.get('shot_fired', False):
            reward -= 0.0 
            
        # Phạt trượt nhẹ (-2.0)
        if step_info.get('arrows_went_out', 0) > 0:
            reward -= 2.0 * step_info['arrows_went_out']
            
        # 2. XỬ LÝ HIT (Quan trọng: Cần Env trả về hit_priority_target)
        if step_info.get('hit_priority_target', False):
            # Bắn trúng ĐÚNG priority target -> Thưởng lớn
            reward += 100.0
        
        # 3. LINEAR SHAPING REWARD (Công thức tuyến tính của bạn)
        active_arrows = obs.get('arrows', [])
        active_targets = obs.get('targets', [])

        # Tính điểm dựa trên khoảng cách (cho các frame mũi tên đang bay)
        if active_arrows and active_targets:
            priority_target = active_targets[0] 
            t_pos = np.array([priority_target['pos']['x'], priority_target['pos']['y']])
            
            min_dist = float('inf')
            for arrow in active_arrows:
                a_pos = np.array([arrow['pos']['x'], arrow['pos']['y']])
                dist = np.linalg.norm(a_pos - t_pos)
                if dist < min_dist:
                    min_dist = dist
            
            # --- CÔNG THỨC TUYẾN TÍNH (1/x) ---
            # K = 200, C = 2.0
            dist_reward = 200.0 / (min_dist + 2.0) 
            
            # Nhân với 0.05 để mỗi frame nhận một chút
            reward += dist_reward * 0.05

        # 4. IDLE PENALTY
        # Phạt nhẹ để ép hành động
        if len(active_arrows) == 0 and not step_info.get('shot_fired', False):
            reward -= 0.1

        return reward