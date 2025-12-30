import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # Tầng dùng chung (Shared layers)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Nhánh Actor: Quyết định hành động (Góc, Lực, Bắn)
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh() # Đưa hành động về khoảng [-1, 1] để dễ chuẩn hóa
        )
        
        # Nhánh Critic: Đánh giá trạng thái (Value)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        action_values = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_values, state_value