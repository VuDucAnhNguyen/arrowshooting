import torch

class Params:
    def __init__ (self):
        self.seed = 42

        self.input_dim = 66      
        self.hidden_dim = 256   # Số neuron lớp ẩn
        self.output_dim = 3     # Action: mean(angle), mean(power)
        
        self.lr = 3e-4             # Tốc độ học 
        self.gamma = 0.99          # Discount factor (trọng số tương lai)
        self.beta = 0.001           # hệ số entropy
        self.eps_clip = 0.2        # PPO cần thêm cái này (tỉ lệ cắt)
        self.K_epochs = 10         # Số lần học lại trên 1 batch (cho PPO)
        self.gae_lambda = 0.95
        
        self.training_num_episodes = 2000   # Tổng số màn chơi để train
        self.n_steps = 2048
        self.batch_size = 64
        
        self.save_path = "arrow_shooting_ppo_best.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = Params() 