"""
Neural Network Architecture với Attention Mechanism
Thiết kế đặc biệt để xử lý multiple targets và temporal dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from params import Config


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention cho multiple targets
    
    Giúp agent focus vào targets quan trọng nhất
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) - True for valid positions
        
        Returns:
            attended: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq, 3*embed)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attended)


class TargetEncoder(nn.Module):
    """
    Encode target information with attention
    
    Input: Multiple targets với varying positions/velocities
    Output: Fixed-size representation of all targets
    """
    
    def __init__(
        self, 
        target_feature_dim: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        max_targets: int = 5
    ):
        super().__init__()
        
        self.target_feature_dim = target_feature_dim
        self.embed_dim = embed_dim
        self.max_targets = max_targets
        
        # Project target features to embedding
        self.target_embed = nn.Sequential(
            nn.Linear(target_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Attention over targets
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, target_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_features: (batch, max_targets, feature_dim)
            mask: (batch, max_targets) - True for valid targets
        
        Returns:
            target_repr: (batch, embed_dim) - Aggregated target representation
        """
        # Embed targets
        embedded = self.target_embed(target_features)  # (batch, max_targets, embed_dim)
        
        # Apply attention
        attended = self.attention(embedded, mask)  # (batch, max_targets, embed_dim)
        
        # Aggregate (mean pooling over valid targets)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, max_targets, 1)
        summed = (attended * mask_expanded).sum(dim=1)  # (batch, embed_dim)
        count = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        aggregated = summed / count
        
        return self.output_proj(aggregated)


class PhysicsPredictor(nn.Module):
    """
    Predict arrow trajectory given current state
    
    Auxiliary task để improve representation learning
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predict (x, y) after N steps
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, input_dim)
        
        Returns:
            predicted_pos: (batch, 2) - Predicted (x, y)
        """
        return self.predictor(features)


class ArrowFeatureExtractor(BaseFeaturesExtractor):
    """
    Advanced feature extractor với attention và physics reasoning
    
    Architecture:
    1. Separate encoders cho player, wind, resources
    2. Target encoder với attention mechanism
    3. Physics predictor (auxiliary task)
    4. Fusion network
    """
    
    def __init__(
        self, 
        observation_space: spaces.Dict,
        config: Config,
        features_dim: int = 512
    ):
        super().__init__(observation_space, features_dim)
        
        self.config = config
        self.sp = config.state
        self.np = config.network
        
        # === COMPONENT ENCODERS ===
        
        # Player encoder (simple)
        self.player_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Wind encoder (with history if enabled)
        wind_input_dim = 2
        if self.sp.include_wind_history:
            wind_input_dim += 2 * self.sp.wind_history_length
        self.wind_encoder = nn.Sequential(
            nn.Linear(wind_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.wind_history_buffer = []
        
        # Resources encoder
        self.resources_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Target encoder with attention
        target_feature_dim = self._compute_target_feature_dim()
        if self.np.use_attention:
            self.target_encoder = TargetEncoder(
                target_feature_dim=target_feature_dim,
                embed_dim=128,
                num_heads=self.np.attention_heads,
                max_targets=self.sp.max_targets_tracked
            )
            target_output_dim = 128
        else:
            # Simple MLP if no attention
            self.target_encoder = nn.Sequential(
                nn.Linear(target_feature_dim * self.sp.max_targets_tracked, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            target_output_dim = 128
        
        # === FUSION NETWORK ===
        fusion_input_dim = 64 + 64 + 64 + target_output_dim  # player + wind + resources + targets
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.LayerNorm(features_dim) if self.np.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim) if self.np.use_layer_norm else nn.Identity(),
            nn.ReLU()
        )
        
        # === AUXILIARY TASKS (optional) ===
        if config.auxiliary.use_trajectory_prediction:
            self.trajectory_predictor = PhysicsPredictor(features_dim, 128)
        
        if config.auxiliary.use_hit_prediction:
            self.hit_predictor = nn.Sequential(
                nn.Linear(features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        if self.np.use_orthogonal_init:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Orthogonal initialization (PPO best practice)"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def _compute_target_feature_dim(self) -> int:
        """Compute target feature dimension based on config"""
        dim = 6  # Base: pos(2) + vel(2) + distance(1) + angle(1)
        
        if self.sp.include_target_prediction:
            dim += 2 * len(self.sp.prediction_horizons)  # Predicted positions
        
        if self.sp.include_target_trajectory:
            dim += 2  # Acceleration estimate
        
        if self.sp.include_interception_point:
            dim += 2  # Optimal interception point
        
        return dim
    
    def forward(self, observations: Dict) -> torch.Tensor:
        """
        Extract features from observations
        
        Args:
            observations: Dict with player, wind, resources, targets
        
        Returns:
            features: (batch, features_dim)
        """
        # Ensure observations are tensors
        observations = self._ensure_tensors(observations)
        
        # === ENCODE COMPONENTS ===
        
        # Player
        player_features = self._encode_player(observations['player'])
        
        # Wind (with history)
        wind_features = self._encode_wind(observations['wind'])
        
        # Resources
        resource_features = self._encode_resources(observations['resources'])
        
        # Targets (with attention)
        target_features, target_mask = self._encode_targets(observations['targets'])
        
        # === FUSE FEATURES ===
        fused = torch.cat([
            player_features,
            wind_features,
            resource_features,
            target_features
        ], dim=-1)
        
        features = self.fusion(fused)
        
        return features
    
    def _ensure_tensors(self, obs: Dict) -> Dict:
        """Convert numpy arrays to tensors if needed"""
        def convert(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            elif isinstance(x, dict):
                return {k: convert(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [convert(item) for item in x]
            else:
                return x
        
        return convert(obs)
    
    def _encode_player(self, player: Dict) -> torch.Tensor:
        """Encode player position"""
        x = player['x'] / self.sp.world_width
        y = player['y'] / self.sp.world_height
        
        # Handle batch dimension
        if x.dim() == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        
        player_vec = torch.stack([x, y], dim=-1)
        return self.player_encoder(player_vec)
    
    def _encode_wind(self, wind: Dict) -> torch.Tensor:
        """Encode wind with history"""
        wx = wind['x'] / 0.2
        wy = wind['y'] / 0.2
        
        if wx.dim() == 0:
            wx = wx.unsqueeze(0)
            wy = wy.unsqueeze(0)
        
        wind_vec = torch.stack([wx, wy], dim=-1)
        
        # Add history if enabled
        if self.sp.include_wind_history:
            # Update buffer
            self.wind_history_buffer.append(wind_vec.detach())
            if len(self.wind_history_buffer) > self.sp.wind_history_length:
                self.wind_history_buffer.pop(0)
            
            # Pad if not enough history
            history = self.wind_history_buffer.copy()
            while len(history) < self.sp.wind_history_length:
                history.insert(0, wind_vec)
            
            # Concatenate
            history_vec = torch.cat(history, dim=-1)
            wind_vec = torch.cat([wind_vec, history_vec], dim=-1)
        
        return self.wind_encoder(wind_vec)
    
    def _encode_resources(self, resources: Dict) -> torch.Tensor:
        """Encode resources"""
        mana = resources['mana'] / self.sp.max_mana
        time = resources['time_left'] / self.sp.max_time
        arrows = resources['arrows_left'] / self.sp.max_arrows
        
        if mana.dim() == 0:
            mana = mana.unsqueeze(0)
            time = time.unsqueeze(0)
            arrows = arrows.unsqueeze(0)
        
        resource_vec = torch.stack([mana, time, arrows], dim=-1)
        return self.resources_encoder(resource_vec)
    
    def _encode_targets(self, targets: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode targets with attention
        
        Returns:
            features: (batch, embed_dim)
            mask: (batch, max_targets)
        """
        # TODO: Implement full target encoding with predictions
        # For now, simplified version
        
        # Placeholder implementation
        batch_size = 1
        max_targets = self.sp.max_targets_tracked
        feature_dim = self._compute_target_feature_dim()
        
        # Create feature matrix and mask
        target_matrix = torch.zeros(batch_size, max_targets, feature_dim)
        mask = torch.zeros(batch_size, max_targets, dtype=torch.bool)
        
        # Fill with actual targets
        for i, target in enumerate(targets[:max_targets]):
            if isinstance(target, dict):
                # Extract features
                tx = target['pos']['x'] / self.sp.world_width
                ty = target['pos']['y'] / self.sp.world_height
                vx = target['vel']['x'] / self.sp.max_velocity
                vy = target['vel']['y'] / self.sp.max_velocity
                
                # TODO: Add more features
                target_matrix[0, i, :6] = torch.tensor([tx, ty, vx, vy, 0.5, 0.5])
                mask[0, i] = True
        
        # Encode with attention
        if self.np.use_attention:
            encoded = self.target_encoder(target_matrix, mask)
        else:
            encoded = self.target_encoder(target_matrix.flatten(1))
        
        return encoded, mask
    
    def predict_trajectory(self, features: torch.Tensor) -> torch.Tensor:
        """Auxiliary task: predict arrow trajectory"""
        if hasattr(self, 'trajectory_predictor'):
            return self.trajectory_predictor(features)
        return None
    
    def predict_hit_probability(self, features: torch.Tensor) -> torch.Tensor:
        """Auxiliary task: predict hit probability"""
        if hasattr(self, 'hit_predictor'):
            return self.hit_predictor(features)
        return None


if __name__ == "__main__":
    from params import get_config
    
    config = get_config()
    print("Testing ArrowFeatureExtractor with attention...")
    
    # Create dummy observation space
    obs_space = spaces.Dict({
        'player': spaces.Dict({'x': spaces.Box(0, 1800), 'y': spaces.Box(0, 900)}),
        'wind': spaces.Dict({'x': spaces.Box(-0.2, 0.2), 'y': spaces.Box(-0.2, 0.2)}),
        'resources': spaces.Dict({
            'mana': spaces.Box(0, 100),
            'time_left': spaces.Box(0, 3000),
            'arrows_left': spaces.Box(0, 20)
        }),
        'targets': spaces.Sequence(spaces.Dict({
            'pos': spaces.Dict({'x': spaces.Box(0, 1800), 'y': spaces.Box(0, 900)}),
            'vel': spaces.Dict({'x': spaces.Box(-50, 50), 'y': spaces.Box(-50, 50)})
        }))
    })
    
    extractor = ArrowFeatureExtractor(obs_space, config, features_dim=512)
    print(f"✓ Feature extractor created")
    print(f"  Features dim: {config.network.features_dim}")
    print(f"  Attention: {config.network.use_attention}")
    print(f"  Attention heads: {config.network.attention_heads}")
