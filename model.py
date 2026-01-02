import torch
import torch.nn as nn
from torch.distributions import Normal
from params import params

class PPO(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPO, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, params.hidden_dim),
            nn.Tanh(),
            nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.Tanh(),
        )

        #actor mean tính giá trị góc bắn và lực trong khoảng [-1, 1]
        self.actor_mean = nn.Sequential(
            nn.Linear(params.hidden_dim, action_dim),
            nn.Tanh()
        )

        #actor log std tính log độ lệch chuẩn, khởi đầu độ lệch chuẩn là 1 (e^0)
        #tính log nhằm loại bỏ dấu âm std
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Linear(params.hidden_dim, 1)

    def forward(self, state):
        x = self.shared(state)
        
        mean = self.actor_mean(x)

        #chuyển log std -> std
        std = torch.exp(self.actor_log_std.expand_as(mean))

        #tạo phân phối chuẩn theo mean và std
        dist = Normal(mean, std)

        value = self.critic(x)

        return dist, value