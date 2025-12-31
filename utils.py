import torch
import matplotlib.pyplot as plt
from params import params

class Utlis:
    def __init__(self):
        pass

    def load_model(self, agent):
        agent.model.load_state_dict(torch.load(params.save_path, map_location=params.device))
        print("Đã load model thành công!")

    def save_model(self, agent):
        torch.save(agent.model.state_dict(), params.save_path)

    def plot_learning_curve(self, raw_rewards, smoothed_rewards):
        plt.figure(figsize=(10, 6))
        plt.plot(raw_rewards, label='Raw Reward (Episode)', color='cyan', alpha=0.3)
        plt.plot(smoothed_rewards, label='Smoothed Reward (Trend)', color='orange', linewidth=2)
        plt.title(f"Training Learning Curve ({params.n_steps}-steps)")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Total Reward per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"result/training_curve_{params.n_steps}steps.png")
        plt.show()

utils = Utlis()