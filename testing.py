import torch
from gymnasium.wrappers import RecordVideo
from params import params
from utils import utils
import numpy as np

class Testing():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def start_testing(self):
        #load model từ file
        try:
            utils.load_model(agent = self.agent)
        except:
            print(f"Chưa có file model tại {params.save_path}")
            return
        
        self.agent.model.eval() # Chuyển sang chế độ test
        all_rewards = []

        num_test = params.test_episode
        for i in range(0,num_test):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
             self.env.render()
             # Chuyển state sang tensor
             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(params.device)
                
             # Chỉ cần lấy action từ model, không cần tính toán loss
             with torch.no_grad():
                dist= self.agent.model.act(state_tensor)
                action = dist.mean

                action = action.cpu().numpy().flatten()
                
             state, reward, terminated, truncated, _ = self.env.step(action)
             done = terminated or truncated
             total_reward += reward
                
            all_rewards.append(total_reward)
            print(f"Episode {i+1}: Reward = {total_reward:.2f}")

        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        print("-" * 30)
        print(f"Result After {num_test} Game:")
        print(f"Mean: {mean_reward:.2f}")
        print(f"Standard deviation: {std_reward:.2f}")
        print(f"Max reward: {np.max(all_rewards)}")
        print("-" * 30)   

        self.env.close()