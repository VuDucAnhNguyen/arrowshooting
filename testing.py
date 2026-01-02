import torch
from gymnasium.wrappers import RecordVideo
from params import params
from utils import utils

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

        state, _ = self.env.reset()
        done = False
        total_reward = 0
            
        while not done:
            self.env.render()
            # Chuyển state sang tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(params.device)
                
            # Chỉ cần lấy action từ model, không cần tính toán loss
            with torch.no_grad():
                dist, _ = self.agent.model(state_tensor)
                action = dist.mean

                action = action.cpu().numpy().flatten()
                
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
                
        print(f"Điểm số = {total_reward}")

            
        self.env.close()