import torch
import numpy as np
import os
from params import params
from utils import utils 

class Training:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        
        # Buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.masks = []
        
        self.time_step = 0
        
        # Danh sách lưu điểm để vẽ sau này
        self.score_history = []    
        self.avg_score_history = []
        
    def start_training(self):
        print(f"🚀 Training Started on {params.device}...")
        
        best_score = -float('inf')
        running_score = 0
        
        # --- VÒNG LẶP TRAINING (2000 EPISODES) ---
        for i_episode in range(1, params.training_num_episodes + 1):
            
            state, _ = self.env.reset()
            done = False
            real_score = 0 # Điểm game thực tế
            
            while not done:
                self.time_step += 1
                
                # 1. Select Action
                action_dict = self.agent.select_action(state)
                action = action_dict['action']
                log_prob = action_dict['log_prob']
                
                # 2. Step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 3. Store
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(log_prob)
                self.rewards.append(reward)
                self.masks.append(0 if done else 1)
                
                state = next_state
                
                # 4. Update PPO
                if self.time_step % params.n_steps == 0:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(params.device)
                    with torch.no_grad():
                        _, next_value = self.agent.model(next_state_tensor)
                        next_state_value = next_value.item()
                    
                    self.agent.update_model(self.states, self.actions, self.log_probs, 
                                          self.rewards, self.masks, next_state_value)
                    
                    self.states.clear(); self.actions.clear(); self.log_probs.clear()
                    self.rewards.clear(); self.masks.clear()

            # --- HẾT 1 VÁN ---
            real_score = info.get('score', 0)
            self.score_history.append(real_score)
            
            # Tính làm mượt luôn để lát nữa vẽ
            avg_window = 50
            if len(self.score_history) >= avg_window:
                smoothed_val = np.mean(self.score_history[-avg_window:])
            else:
                smoothed_val = np.mean(self.score_history)
            self.avg_score_history.append(smoothed_val)
            
            running_score += real_score
            
            # --- LOGGING (Mỗi 10 ván) ---
            if i_episode % 10 == 0:
                avg_10_episodes = running_score / 10
                print(f"Episode: {i_episode} | Avg Score: {avg_10_episodes:.1f} | Best: {best_score:.1f}")
                
                if avg_10_episodes > best_score:
                    best_score = avg_10_episodes
                    utils.save_model(self.agent)
                
                running_score = 0
                # KHÔNG GỌI VẼ Ở ĐÂY NỮA

        self.env.close()
        
        # Gọi vẽ 1 lần duy nhất tại đây
        utils.plot_learning_curve(self.score_history, self.avg_score_history)