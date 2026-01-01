import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from arrow_env import ArrowEnv
from network import ActorCriticNetwork

# -------------------------
# 2. Observation processing
# -------------------------
def flatten_obs(obs):
    """Chuyển observation dict thành vector float"""
    vec = []
    # player
    vec.append(obs["player"]["x"])
    vec.append(obs["player"]["y"])
    # wind
    vec.append(obs["wind"]["x"])
    vec.append(obs["wind"]["y"])
    # resources
    vec.append(obs["resources"]["mana"])
    vec.append(obs["resources"]["time_left"])
    vec.append(obs["resources"]["arrows_left"])
    # first 5 targets (x, y)
    for i in range(5):
        if i < len(obs["targets"]):
            vec.append(obs["targets"][i]["pos"]["x"])
            vec.append(obs["targets"][i]["pos"]["y"])
        else:
            vec.append(0.0)
            vec.append(0.0)
    return np.array(vec, dtype=np.float32)

# -------------------------
# 3. Training Loop
# -------------------------
env = ArrowEnv(render_mode=None)  # render_mode="human" nếu muốn xem game
obs_example, _ = env.reset()
obs_dim = flatten_obs(obs_example).shape[0]
action_dim = 3  # [angle, power, shoot]

network = ActorCriticNetwork(obs_dim, action_dim)
optimizer = optim.Adam(network.parameters(), lr=3e-4)
gamma = 0.99
num_episodes = 1000

for episode in range(num_episodes):
    obs, _ = env.reset()
    obs_vec = flatten_obs(obs)
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32)
        action_mean, state_value = network(obs_tensor)

        # sample action với Gaussian noise
        dist = Normal(action_mean, torch.tensor([5.0, 5.0, 0.5]))
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # clamp action vào range hợp lệ của môi trường
        action_np = action.detach().numpy()
        action_np[0] = np.clip(action_np[0], 0, 90)   # angle
        action_np[1] = np.clip(action_np[1], 10, 50)  # power
        action_np[2] = np.clip(action_np[2], 0, 1)    # shoot

        next_obs, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        # reward: mình dùng reward tạm là score tăng
        reward = info['step_info']['score_gained']
        total_reward += reward

        # compute advantage
        next_obs_vec = flatten_obs(next_obs)
        with torch.no_grad():
            _, next_state_value = network(torch.tensor(next_obs_vec, dtype=torch.float32))
        target_value = reward + gamma * next_state_value * (1 - int(done))
        advantage = target_value - state_value

        # actor + critic loss
        actor_loss = -log_prob * advantage.detach()
        critic_loss = nn.functional.mse_loss(state_value, target_value.detach())
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs_vec = next_obs_vec

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")
