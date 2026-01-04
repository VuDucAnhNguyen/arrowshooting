"""
Hyperparameters cho ArrowShooting DRL Agent
Sử dụng PPO với Curriculum Learning - Chiến lược tối ưu cho sparse reward
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainingParams:
    """Tham số training - Tối ưu cho PPO"""
    
    # PPO specific
    algorithm: str = "PPO"              # PPO ổn định hơn SAC cho sparse reward
    total_timesteps: int = 1_000_000    # Tăng lên để học kỹ
    n_steps: int = 2048                 # Steps mỗi rollout (PPO buffer size)
    batch_size: int = 64                # Mini-batch size
    n_epochs: int = 10                  # Số epochs train mỗi rollout
    
    # Learning rates
    learning_rate: float = 3e-4         # LR ban đầu
    use_lr_schedule: bool = True        # Linear decay
    
    # Discount & GAE
    gamma: float = 0.995                # Discount cao để care về future (targets xa)
    gae_lambda: float = 0.95            # GAE lambda
    
    # PPO clipping
    clip_range: float = 0.2             # PPO clip range
    clip_range_vf: Optional[float] = None  # Value function clip (None = no clip)
    
    # Entropy bonus (exploration)
    ent_coef: float = 0.01              # Entropy coefficient
    ent_coef_decay: bool = True         # Decay entropy over time
    ent_coef_final: float = 0.001       # Final entropy coef
    
    # Value function
    vf_coef: float = 0.5                # Value function coefficient
    max_grad_norm: float = 0.5          # Gradient clipping
    
    # Environment
    n_envs: int = 8                     # Parallel environments (tăng để sample nhiều)
    max_episode_steps: int = 3000       # Max steps per episode
    
    # Logging & saving
    log_interval: int = 1               # Log mỗi 1 update
    eval_freq: int = 10_000             # Eval mỗi 10k steps
    n_eval_episodes: int = 10           # Số episodes eval
    save_freq: int = 50_000             # Save mỗi 50k steps
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    best_model_path: str = "./best_model"
    logs_dir: str = "./logs"
    tensorboard_log: str = "./tensorboard"


@dataclass
class CurriculumParams:
    """Curriculum Learning - Train từ dễ → khó"""
    
    enable_curriculum: bool = True
    
    # Stage 1: Static targets (học aim cơ bản)
    stage1_duration: int = 200_000      # 200k steps
    stage1_target_speed: float = 0.0    # Targets đứng yên
    stage1_wind_strength: float = 0.0   # Không có gió
    
    # Stage 2: Slow moving targets
    stage2_duration: int = 300_000      # 300k steps
    stage2_target_speed: float = 1.0    # Targets di chuyển chậm
    stage2_wind_strength: float = 0.1   # Gió nhẹ
    
    # Stage 3: Full difficulty
    stage3_target_speed: float = 4.0    # Targets nhanh (như env gốc)
    stage3_wind_strength: float = 0.2   # Gió mạnh
    
    # Transition smoothing
    smooth_transition: bool = True      # Transition mượt giữa stages
    transition_steps: int = 20_000      # Số steps để transition


@dataclass
class RewardParams:
    """Reward Shaping - Thiết kế reward thông minh"""
    
    # === SPARSE REWARDS (Main objectives) ===
    hit_reward: float = 500.0           # Thưởng LỚN khi trúng target
    completion_bonus: float = 1000.0    # Bonus khi clear all targets
    arrow_efficiency_bonus: float = 50.0  # Bonus mỗi arrow còn thừa
    
    # === DENSE REWARDS (Shaping - Guide learning) ===
    # Distance-based shaping
    use_distance_shaping: bool = True
    distance_improvement_scale: float = 0.5  # Thưởng khi arrow bay gần target
    closest_approach_reward: float = 10.0    # Thưởng khi arrow đi qua gần target nhất
    
    # Prediction reward (khuyến khích aim đúng hướng)
    aim_quality_reward: float = 5.0     # Thưởng khi aim về hướng target
    trajectory_reward: float = 2.0      # Thưởng khi quỹ đạo hợp lý
    
    # === PENALTIES (Negative shaping) ===
    miss_penalty: float = -10.0         # Phạt khi arrow ra ngoài màn hình
    time_penalty: float = -0.05         # Phạt nhẹ mỗi step (motivate nhanh)
    wasted_shot_penalty: float = -15.0  # Phạt NẶNG khi bắn khi không nên
    low_mana_shot_penalty: float = -5.0 # Phạt khi bắn với mana thấp
    
    # === ADAPTIVE REWARDS (Thay đổi theo training progress) ===
    use_adaptive_rewards: bool = True
    sparse_reward_weight_start: float = 0.3   # Ban đầu ưu tiên dense rewards
    sparse_reward_weight_end: float = 1.0     # Cuối ưu tiên sparse rewards
    adaptive_schedule_steps: int = 500_000    # Transition trong 500k steps
    
    # === RESOURCE MANAGEMENT ===
    mana_threshold: float = 35.0        # Mana tối thiểu để bắn
    critical_arrows_threshold: int = 5  # Cảnh báo khi ít arrows
    resource_panic_penalty: float = -20.0  # Phạt khi waste resources cuối game


@dataclass
class StateParams:
    """State Representation - Feature engineering"""
    
    # Normalization
    world_width: float = 1800.0
    world_height: float = 900.0
    max_velocity: float = 50.0
    max_mana: float = 100.0
    max_arrows: int = 20
    max_time: int = 3000
    
    # === ENGINEERED FEATURES ===
    
    # Wind features
    include_wind_history: bool = True
    wind_history_length: int = 10       # Lưu 10 steps gió gần nhất
    include_wind_prediction: bool = True  # Dự đoán wind trend
    
    # Target features
    include_target_prediction: bool = True
    prediction_horizons: Tuple[int, ...] = (10, 20, 30, 50)  # Dự đoán nhiều horizons
    include_target_trajectory: bool = True  # Velocity + acceleration
    include_interception_point: bool = True  # Điểm giao nhau tối ưu
    
    # Arrow features (khi đang có arrows bay)
    include_arrow_trajectory: bool = True
    max_arrows_tracked: int = 3         # Track 3 arrows gần nhất
    
    # Geometric features
    include_angles_distances: bool = True  # Góc và khoảng cách đến targets
    include_relative_positions: bool = True  # Vị trí tương đối
    
    # Physics simulation features
    simulate_trajectories: bool = True   # Simulate arrow trajectory với current params
    n_trajectory_points: int = 20       # Số điểm simulate
    
    # === STATE AUGMENTATION ===
    use_state_normalization: bool = True  # Normalize states
    use_state_stacking: bool = False     # Frame stacking (không cần vì có history)


@dataclass
class NetworkParams:
    """Neural Network Architecture"""
    
    # Feature extractor
    features_dim: int = 512             # Output dim của feature extractor (tăng lên)
    
    # Policy network (Actor)
    policy_layers: Tuple[int, ...] = (512, 512, 256)  # 3 layers sâu hơn
    policy_activation: str = "tanh"     # tanh hoặc relu
    
    # Value network (Critic)
    value_layers: Tuple[int, ...] = (512, 512, 256)
    value_activation: str = "relu"
    
    # Shared network (optional)
    use_shared_network: bool = False    # Share feature extractor giữa actor-critic
    
    # Advanced features
    use_attention: bool = True          # Attention mechanism cho targets
    attention_heads: int = 4            # Multi-head attention
    use_lstm: bool = False              # LSTM cho temporal dependencies (nặng, cân nhắc)
    lstm_hidden_size: int = 256
    
    # Normalization
    use_layer_norm: bool = True         # Layer normalization
    use_orthogonal_init: bool = True    # Orthogonal initialization (PPO best practice)


@dataclass
class ExplorationParams:
    """Exploration Strategy"""
    
    # Action noise (cho continuous actions)
    use_action_noise: bool = True
    action_noise_type: str = "normal"   # "normal" or "ornstein_uhlenbeck"
    action_noise_std: float = 0.1       # Std của noise
    
    # Exploration schedule
    noise_decay: bool = True            # Giảm noise theo thời gian
    noise_min: float = 0.01             # Noise tối thiểu
    
    # Epsilon-greedy for shooting decision
    use_epsilon_shoot: bool = True      # Epsilon cho quyết định có bắn không
    epsilon_start: float = 0.3          # Explore 30% ban đầu
    epsilon_end: float = 0.05           # 5% cuối
    epsilon_decay_steps: int = 300_000


@dataclass
class AuxiliaryTaskParams:
    """Auxiliary Tasks - Improve representation learning"""
    
    # Trajectory prediction task
    use_trajectory_prediction: bool = True
    trajectory_pred_weight: float = 0.1     # Weight trong loss function
    
    # Hit probability prediction
    use_hit_prediction: bool = True
    hit_pred_weight: float = 0.1
    
    # Next target position prediction
    use_next_target_pred: bool = True
    next_target_weight: float = 0.05
    
    # Inverse dynamics (predict action from states)
    use_inverse_dynamics: bool = False  # Optional, thường không cần
    inverse_dynamics_weight: float = 0.05


@dataclass
class RuleBasedParams:
    """Rule-based Assistant - Hỗ trợ DRL agent"""
    
    enable_rules: bool = True
    
    # Shooting rules
    min_mana_to_shoot: float = 35.0         # Không bắn nếu mana < 35
    max_arrows_in_flight: int = 2           # Không bắn nếu đã có 2 arrows bay
    wait_for_mana_regen: bool = True        # Đợi mana hồi nếu thấp
    
    # Target selection rules
    prioritize_closest: bool = True         # Ưu tiên target gần
    prioritize_easier_angle: bool = True    # Ưu tiên target góc dễ bắn
    avoid_difficult_shots: bool = True      # Không bắn shots khó (gió mạnh)
    
    # Physics constraints
    enforce_min_power: float = 15.0         # Power tối thiểu
    enforce_max_power: float = 48.0         # Power tối đa
    enforce_angle_range: Tuple[float, float] = (15.0, 75.0)  # Góc hợp lý
    
    # Decision override (when to let rules decide)
    override_on_low_arrows: bool = True     # Rules quyết định khi ít arrows
    override_threshold_arrows: int = 3      # Threshold để override


@dataclass
class Config:
    """Master Configuration"""
    training: TrainingParams = TrainingParams()
    curriculum: CurriculumParams = CurriculumParams()
    reward: RewardParams = RewardParams()
    state: StateParams = StateParams()
    network: NetworkParams = NetworkParams()
    exploration: ExplorationParams = ExplorationParams()
    auxiliary: AuxiliaryTaskParams = AuxiliaryTaskParams()
    rules: RuleBasedParams = RuleBasedParams()
    
    def __post_init__(self):
        """Validation"""
        assert self.training.n_steps % self.training.batch_size == 0
        assert self.reward.hit_reward > 0
        assert 0 < self.training.gamma < 1
        print("✓ Configuration validated")


def get_config() -> Config:
    """Get default config"""
    return Config()


def print_config(config: Config):
    """Pretty print configuration"""
    print("\n" + "="*80)
    print("ARROWSHOOTING DRL AGENT - OPTIMIZED CONFIGURATION")
    print("="*80)
    
    sections = [
        ("TRAINING (PPO)", config.training),
        ("CURRICULUM LEARNING", config.curriculum),
        ("REWARD SHAPING", config.reward),
        ("STATE ENGINEERING", config.state),
        ("NETWORK ARCHITECTURE", config.network),
        ("EXPLORATION", config.exploration),
        ("AUXILIARY TASKS", config.auxiliary),
        ("RULE-BASED ASSIST", config.rules)
    ]
    
    for title, params in sections:
        print(f"\n[{title}]")
        for key, value in params.__dict__.items():
            if isinstance(value, tuple) and len(value) > 3:
                value = f"({value[0]}, ..., {value[-1]})"
            print(f"  {key:.<35} {value}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("  • Using PPO for stability with sparse rewards")
    print("  • Curriculum learning: static → moving targets")
    print("  • Dense reward shaping to guide early learning")
    print("  • Attention mechanism for multi-target handling")
    print("  • Auxiliary tasks for better representations")
    print("="*80 + "\n")


if __name__ == "__main__":
    config = get_config()
    print_config(config)