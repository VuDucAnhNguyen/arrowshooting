from stable_baselines3 import PPO

# 1. Khởi tạo môi trường (bật render_mode="human" để xem)
env = make_env() # Sử dụng hàm make_env bạn đã viết

# 2. Tải model từ file .zip
model = PPO.load("ppo_arrow_final_model")

# 3. Chạy vòng lặp cho AI bắn
obs, info = env.reset()
while True:
    # deterministic=True giúp AI chọn hành động tốt nhất (không bắn thử nghiệm nữa)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render() # Vẽ hình ảnh
    
    if terminated or truncated:
        obs, info = env.reset()