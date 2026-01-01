# --- Testing ---
num_test_episodes = 10

for episode in range(num_test_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_mean, _ = network(obs_tensor)
            action = action_mean.numpy()  # Chọn hành động "deterministic" để test
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = next_obs
        
    print(f"[TEST] Episode {episode+1}, Total Reward: {total_reward:.2f}")
