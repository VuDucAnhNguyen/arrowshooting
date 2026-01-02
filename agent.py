# agent.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import PPO
from params import params

class PPOAgent:
    def __init__(self, input_dim, action_dim):
        self.model = PPO(input_dim, action_dim).to(params.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr)
        self.MseLoss = nn.MSELoss()

    # lấy action (chuẩn hóa nó)
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(params.device)
        
        with torch.no_grad():
            dist = self.model.act(state_tensor)
            value = self.model.critic(state_tensor)

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return {
            'action': action.cpu().numpy().flatten(), 
            'log_prob': log_prob.item(),
            'value': value.item()
        }

    def update_model(self, states, actions, log_probs, rewards, masks, values, next_state_value):
        states = torch.FloatTensor(np.array(states)).to(params.device)
        actions = torch.FloatTensor(np.array(actions)).to(params.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(params.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(params.device)
        masks = torch.FloatTensor(np.array(masks)).to(params.device)
        values = torch.FloatTensor(np.array(values)).to(params.device)

        returns = []
        advantages = []

        with torch.no_grad():
            # Tính lại values cho chính xác (hoặc dùng values đã lưu)
            _, current_values = self.model.evaluate(states)
            current_values = current_values.squeeze()
            
            # Tính Returns và Advantages
            returns = torch.zeros_like(rewards).to(params.device)
            advantages = torch.zeros_like(rewards).to(params.device)
            
            running_return = 0
            previous_value = 0
            running_advantage = 0
            
            # Tính ngược từ cuối lên
            for t in reversed(range(len(rewards))):
                # 1. Tính Return (Reward-to-go) cho Critic học
                running_return = rewards[t] + params.gamma * running_return * masks[t]
                returns[t] = running_return
                
                # 2. Tính GAE cho Actor học (Ổn định hơn nhiều)
                # Nếu là bước cuối cùng
                if t == len(rewards) - 1:
                    next_val = next_state_value 
                else:
                    # Nếu không, lấy value của bước kế tiếp trong batch
                    next_val = values[t + 1]
                    
                delta = rewards[t] + params.gamma * next_val * masks[t] - current_values[t]
                running_advantage = delta + params.gamma * 0.95 * running_advantage * masks[t] # 0.95 là GAE lambda
                advantages[t] = running_advantage

        # CHUẨN HÓA ADVANTAGE 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        loss_value = 0
        
        for _ in range(params.K_epochs):
            # Evaluate lại state cũ để lấy log_prob mới và value mới
            dist, state_values = self.model.evaluate(states)
            state_values = state_values.squeeze()
            
            # Lấy log_prob của hành động cũ trên phân phối mới
            curr_log_probs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            
            # Tính tỷ lệ Ratio (pi_new / pi_old)
            ratios = torch.exp(curr_log_probs - old_log_probs)

            # Tính Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - params.eps_clip, 1 + params.eps_clip) * advantages
            
            # Loss tổng hợp
            # Actor loss: -min(...)
            # Critic loss: MSE(new_value, returns)
            # Entropy bonus: trừ đi để khuyến khích explore (hoặc cộng vào loss âm)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Gradient Clipping (Tránh nổ gradient)
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            
            loss_value = loss.mean().item()
            
        return loss_value

