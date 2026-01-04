import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=128, output_dim=3):
        """
        input_dim: số lượng features sau khi flatten observation
        output_dim: 3 action -> [angle, power, shoot]
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        x: tensor shape (batch_size, input_dim)
        returns: action tensor (angle, power, shoot)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output raw, sẽ map sau để vào đúng action space
        x = self.fc3(x)
        
        # Map outputs vào đúng giới hạn environment
        angle = torch.clamp(x[:, 0], 0, 90)           # 0-90 degrees
        power = torch.clamp(x[:, 1], 10, 50)          # 10-50 units
        shoot = torch.sigmoid(x[:, 2])                # 0-1 probability
        
        return torch.stack([angle, power, shoot], dim=1)
