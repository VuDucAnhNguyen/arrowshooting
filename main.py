from params import params
from agent import PPOAgent
from training import Training
from testing import Testing
from arrow_env import ArrowEnv

from reward_wrapper import RewardWrapper
from observation_wrapper import ObsWrapper
from action_wrapper import ArrowActionWrapper

class Main:
    def __init__(self, mode):
        self.mode = mode
        self.agent = PPOAgent(input_dim = params.input_dim, action_dim = params.output_dim)

        if (mode == 0):
            self.env = self.create_env(render_mode = None)
            self.trainer = Training(agent=self.agent, env=self.env)
        else:
            self.env = self.create_env(render_mode = "rgb_array")
            self.tester = Testing(agent=self.agent, env=self.env)

    def create_env (self, render_mode):
        env = ArrowEnv(render_mode = render_mode)
        env = RewardWrapper(env)
        env = ArrowActionWrapper(env) # Bọc thêm ActionWrapper ở đây
        env = ObsWrapper(env)
        return env

    def run(self):
        if (self.mode == 0):
            self.trainer.start_training()
        else:
            self.tester.start_testing()


if (__name__ == "__main__"):
    try:
        user_input = input("Mode: Training(0), Test(1): ")
        mode = int(user_input)
        if (mode not in [0, 1]):
            print("Invalid input, default mode: Training")
            mode = 0
    except:
        print("Invalid input, default mode: Training")
        mode = 0

    PPO = Main(mode = mode)
    PPO.run()