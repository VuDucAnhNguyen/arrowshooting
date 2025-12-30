from stable_baselines3 import PPO
import os

class ArrowAgent:
    def __init__(self, env, model_path=None):
        self.env = env
        if model_path and os.path.exists(model_path + ".zip"):
            # Load agent đã có sẵn dữ liệu training
            self.model = PPO.load(model_path, env=self.env)
            print(f"--- Đã load dữ liệu training từ {model_path} ---")
        else:
            # Tạo agent mới hoàn toàn
            self.model = PPO(
                policy="MlpPolicy", 
                env=self.env, 
                verbose=1,
                learning_rate=0.0003,
                tensorboard_log="./logs/"
            )
            print("--- Khởi tạo Agent mới ---")

    def train(self, total_timesteps=100000, callback=None):
        print(f"Đang bắt đầu học {total_timesteps} bước...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.save("models/arrow_champion")

    def predict(self, obs):
        # Dùng khi thi đấu: Trả về hành động từ observation
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path):
        # Lưu dữ liệu training vào file .zip
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Đã lưu não bộ Agent tại: {path}.zip")