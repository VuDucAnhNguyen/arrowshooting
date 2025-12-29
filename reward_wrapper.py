import gymnasium as gym

class RewardWrapper(gym.Wrapper):
    
    def __init__ (self, env):
        super().__init__(env)

    def step(self, action):
        obs, terminated, truncated, info = self.env.step(action)

        reward = self.calculate_reward(info)

        return obs, reward, terminated, truncated, info
    

    def calculate_reward(self, info):
        reward = 0.0

        step_info = info.get('step_info', {})

        hit_in_step = step_info.get('targets_hit', 0)
        miss_in_step = step_info.get('arrows_went_out', 0)
        shot_fired = step_info.get('shot_fired', False)

        total_missed_so_far = info.get('arrows_missed', 0)
        arrows_left = info.get('arrows_left', 0)
        arrows_active = info.get('arrows_active', 0)
        active_targets = info.get('active_targets', 0)


        if shot_fired:
            reward -= 5.0

        if hit_in_step > 0:
            reward += 100 * hit_in_step
            reward += (arrows_left + arrows_active) * 5.0
        
        if miss_in_step > 0:
            reward -=  50 * miss_in_step
            reward -= (total_missed_so_far * 2)

        if arrows_left == 0 and active_targets > 0:
            reward -= 200

        return reward