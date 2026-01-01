from params import params
from utils import utils
import torch
from reward_wrapper import RewardWrapper 
from observation_wrapper import ObsWrapper 

class Training:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def start_training(self):
        self.agent.train(total_timesteps=params.n_steps * params.training_num_episodes)