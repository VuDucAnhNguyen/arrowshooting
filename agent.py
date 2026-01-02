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
            dist, value = self.model(state_tensor)

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return {
            'action': action.cpu().numpy().flatten(), 
            'log_prob': log_prob.item(),
            'value': value.item()
        }

    def update_model(self, states, actions, log_probs, rewards, masks, next_state_value):
        old_states = torch.FloatTensor(np.array(states)).to(params.device)
        old_actions = torch.FloatTensor(np.array(actions)).to(params.device)
        old_logprobs = torch.FloatTensor(np.array(log_probs)).to(params.device)

        returns = []
        R = next_state_value
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + params.gamma * R * masks[step]
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(params.device)
        #chuẩn hóa: (z-mean)/std, 1e-7 để tránh trường hợp độ lệch chuẩn thành 0
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        loss_value = 0
        for _ in range(params.K_epochs):
            dist, state_values = self.model(old_states)
            state_values = state_values.squeeze()
            
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - params.eps_clip, 1 + params.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - params.beta * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            loss_value = loss.mean().item()
            
        return loss_value
    

