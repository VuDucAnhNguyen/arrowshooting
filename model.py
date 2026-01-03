import torch
import torch.nn as nn
from torch.distributions import Normal
from params import params

class PPO(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPO, self).__init__()

        # 1. CRITIC NETWORK (Đánh giá trạng thái)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 2. ACTOR NETWORK (Quyết định hành động)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Output [-1, 1]
        )

        #actor log std tính log độ lệch chuẩn, khởi đầu độ lệch chuẩn là 1 (e^0)
        #tính log nhằm loại bỏ dấu âm std
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        

    def forward(self, state):
        raise NotImplementedError
    
    def act(self, state):
        """Dùng khi lấy action để chơi"""
        action_mean = self.actor(state)
        action_log_std = torch.clamp(
            self.actor_log_std.expand_as(action_mean), -2, 1
        )
        action_std = torch.exp(action_log_std)
        
        dist = Normal(action_mean, action_std)
        
        return dist

    def evaluate(self, state):
        """Dùng khi update model (trả về cả dist và value)"""
        action_mean = self.actor(state)
        action_log_std = torch.clamp(
            self.actor_log_std.expand_as(action_mean), -2, 1
        )
        action_std = torch.exp(action_log_std)
        
        dist = Normal(action_mean, action_std)
        state_value = self.critic(state)
        
        return dist, state_value