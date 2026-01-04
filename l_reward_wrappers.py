import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.min_dist_ever = 2000.0

    def reset(self, **kwargs):
        self.min_dist_ever = 2000.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, terminated, truncated, info = self.env.step(action)
        arrows = obs.get('arrows', [])
        targets = obs.get('targets', [])

        reward = 0.0
        reward -= 0.01

# Bước 3: Tính toán khoảng cách (KHÔNG DÙNG ID)
        if len(arrows) > 0 and len(targets) > 0:
            current_min_dist = 2000.0
            
            for arrow in arrows:
                # Lấy tọa độ x, y từ dictionary pos của mũi tên
                a_pos = np.array([arrow['pos']['x'], arrow['pos']['y']])
                
                for target in targets:
                    # Lấy tọa độ x, y từ dictionary pos của mục tiêu
                    t_pos = np.array([target['pos']['x'], target['pos']['y']])
                    
                    # Tính khoảng cách thực tế giữa mũi tên này và mục tiêu này
                    dist = np.linalg.norm(a_pos - t_pos)
                    
                    if dist < current_min_dist:
                        current_min_dist = dist
            
            # THƯỞNG TIẾN TRIỂN: Nếu mũi tên bay lại gần mục tiêu hơn bước trước
            if current_min_dist < self.prev_min_dist:
                # Giải quyết Credit Assignment: Thưởng cho hành động dẫn đến việc bay gần đích
                custom_reward += (self.prev_min_dist - current_min_dist) * 0.5
            
            # Cập nhật lại khoảng cách để dùng cho step sau
            self.prev_min_dist = current_min_dist
        else:
            # Nếu không có mũi tên, đặt lại khoảng cách lớn để chờ lần bắn sau
            self.prev_min_dist = 2000.0

        step_info = info['step_info']


        reward -= step_info["shot_fired"] * 0 # trừ 2 mỗi lần bắn
        reward += step_info["targets_hit"] * 200 # +100 với mỗi target bắn trúng
        reward -= step_info["arrows_went_out"] * 0 # -2 với mỗi viên đạn bắn

        if terminated:
            reward += 250
            reward += info["arrows_left"]
        if truncated:
            reward -= 10

        return obs, reward, terminated, truncated, info

# reward là một trong những yếu tố để update policy