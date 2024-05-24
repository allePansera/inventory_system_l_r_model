from stable_baselines3 import DQN
from agents.AgentAbs import Agent
from analysis import RewardCallback
import gym


class DqnMlp(Agent):
    """
    TODO: Insert here all agent descriptors
    """
    def __init__(self, w_env: gym.Env):
        """
        :param w_env: gym Environment instance
        """
        self.w_env = w_env
        self.model = DQN('MlpPolicy', self.w_env, verbose=1)
        self.reward_callback = RewardCallback()

    def train(self, episode_duration=1000, plot_rewards=True):
        """
        Train the agent.
        :param episode_duration: how many episode run to find optimal policy and value function
        :param plot_rewards: choose whether plot or not reward progression during training
        :return:
        """
        # Train the agent for at least 10.000 months
        self.model.learn(total_timesteps=episode_duration, callback=self.reward_callback)
        if plot_rewards:
            self.reward_callback.plot_rewards()

    def predict(self, observation):
        return self.model.predict(observation)

    def save_model(self, path: str = "models/dqn_mlp"):
        self.model.save(path)

    def load_model(self, path: str = "models/dqn_mlp"):
        self.model.load(path)

    def __repr__(self):
        return "DQN - MLP Policy"


