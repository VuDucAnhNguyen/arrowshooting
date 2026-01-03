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
        if len(states) < 2:
            return 0.0

        states = torch.FloatTensor(np.array(states)).to(params.device)
        actions = torch.FloatTensor(np.array(actions)).to(params.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(params.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(params.device)
        masks = torch.FloatTensor(np.array(masks)).to(params.device)
        old_values = torch.FloatTensor(np.array(values)).to(params.device)
        

        advantages = []
        gae = 0

        
        if not isinstance(next_state_value, torch.Tensor):
            next_state_value = torch.tensor(
            next_state_value, device=params.device, dtype=torch.float32
            )
            
        # Tính ngược từ cuối lên
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_state_value
            else:
                next_value = old_values[t + 1]

            delta = rewards[t] + params.gamma * next_value * masks[t] - old_values[t]
            gae = delta + params.gamma * params.gae_lambda * masks[t] * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(params.device)
        returns = advantages + old_values

        # CHUẨN HÓA ADVANTAGE 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        loss_value = 0
        
        for _ in range(params.K_epochs):
            # Evaluate lại state cũ để lấy log_prob mới và value mới
            dist, new_values = self.model.evaluate(states)
            new_values = new_values.squeeze()
            
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
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(new_values, returns) - params.beta * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Tránh nổ gradient)
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            
            loss_value = loss.mean().item()
            
        return loss_value

