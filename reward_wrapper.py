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
        
        # 1. Cơ chế Thưởng/Phạt cơ bản (Hard Events)
        if step_info.get('shot_fired', False):
            reward -= 2.0  # Giảm mức phạt bắn để khuyến khích Agent thử nghiệm
            
        if step_info.get('targets_hit', 0) > 0:
            reward += 150.0 * step_info['targets_hit'] # Thưởng lớn khi trúng
            
        if step_info.get('arrows_went_out', 0) > 0:
            reward -= 10.0 * step_info['arrows_went_out']

        # 2. Distance Reward (Reward Shaping - Linh hồn của sự cải tiến)
        # Mục tiêu: Thưởng khi mũi tên bay lại gần Target
        active_arrows = obs.get('arrows', [])
        active_targets = obs.get('targets', [])

        if active_arrows and active_targets:
            current_min_dist = float('inf')
            
            for arrow in active_arrows:
                a_pos = np.array([arrow['pos']['x'], arrow['pos']['y']])
                for target in active_targets:
                    t_pos = np.array([target['pos']['x'], target['pos']['y']])
                    
                    dist = np.linalg.norm(a_pos - t_pos)
                    if dist < current_min_dist:
                        current_min_dist = dist
            
            # Thưởng "tiệm cận": Khoảng cách càng gần, reward càng tăng nhẹ
            # Công thức: 1 / (khoảng cách + 1) để tránh chia cho 0
            proximity_reward = 10.0 / (current_min_dist / 100.0 + 1.0)
            reward += proximity_reward

        # 3. Phạt "Idle" (Không làm gì)
        # Nếu không có mũi tên nào trên màn hình và Agent không bắn, phạt nhẹ
        if len(active_arrows) == 0 and not step_info.get('shot_fired', False):
            reward -= 0.1

        # 4. Bonus khi hoàn thành màn chơi sớm
        if info.get('active_targets') == 0:
            reward += info.get('arrows_left', 0) * 10.0
            reward += (info.get('time_left', 0) / 100.0) * 5.0

        return reward