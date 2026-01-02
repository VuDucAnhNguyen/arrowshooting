# training.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from params import params
from utils import utils
import torch
import numpy as np

class Training:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def start_training(self):
        self.agent.model.train()
        
        n_episode = params.training_num_episodes
        raw_history = []
        smoothed_history = []
        running_reward = 0

        states = []
        actions = []
        log_probs = []
        rewards = []
        masks = []
        values = []

        for episode in range(1, n_episode + 1):
            state, _ = self.env.reset()
            done = False
            total_rewards = 0
            step_count = 0

            while not done:
                res = self.agent.get_action(state)
                action = res['action']
                log_prob = res['log_prob']
                value = res['value']

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                log_probs.append(float(log_prob)) # Ép kiểu log_prob
                rewards.append(float(reward))    # Ép kiểu reward từ np.float64 -> float
                masks.append(0.0 if terminated else 1.0)
                values.append(value)

                total_rewards += reward
                state = next_state
                step_count += 1

                if step_count % params.n_steps == 0 or done:
                    if done and not truncated:
                        next_state_value = 0.0
                    else:
                        # Ép kiểu state sang float32 tensor trước khi đưa vào model
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(params.device)
                        with torch.no_grad():
                            _, next_val = self.agent.model.evaluate(next_state_tensor)
                            next_state_value = next_val.item()

                    self.agent.update_model(
                        states=states,
                        actions=actions,
                        log_probs=log_probs,
                        rewards=rewards,
                        masks=masks,
                        values=values,
                        next_state_value=next_state_value
                    )

                    states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []

            # Lưu log và plot 
            if episode == 1:
                running_reward = total_rewards
            else:
                running_reward = 0.05 * total_rewards + 0.95 * running_reward

            raw_history.append(total_rewards)
            smoothed_history.append(running_reward)

            if episode % 10 == 0:
                print(f"Episode {episode} \t Raw {total_rewards:.2f} \t Smooth {running_reward:.2f}")

        utils.save_model(self.agent)
        utils.plot_learning_curve(raw_rewards=raw_history, smoothed_rewards=smoothed_history, save_path="result/training_curve.png")
        self.env.close()