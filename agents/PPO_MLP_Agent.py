from stable_baselines3 import PPO
from agents.AgentAbs import Agent
from analysis import RewardCallback
import gymnasium as gym


class PpoMlp(Agent):

    def __init__(self):
        self.model = None
        self.w_env = None
        self.reward_callback = None

    def train(self, w_env: gym.Env, episode_duration=1000, plot_rewards=True):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param episode_duration: how many episode run to find optimal policy and value function
        :param plot_rewards: choose whether plot or not reward progression during training
        :return:
        """
        self.w_env = w_env
        self.model = PPO('MlpPolicy', self.w_env, verbose=1)
        self.reward_callback = RewardCallback("PPO")
        # Train the agent for at least 10.000 months
        self.model.learn(total_timesteps=episode_duration,
                         callback=self.reward_callback)
        if plot_rewards:
            self.reward_callback.plot_rewards()

    def predict(self, observation):
        """
        Need to call train() before that
        :param observation: list of all states (actually 6)
        :return: action and new states
        """
        assert self.model is not None
        return self.model.predict(observation)

    def save_model(self, path: str = "models/ppo_mlp"):
        """
        Save trained model
        :param path: location to use to read model paramters
        :return:
        """
        assert self.model is not None
        self.model.save(path)

    def load_model(self,  w_env: gym.Env, path: str = "models/ppo_mlp"):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param path: location to use to read model paramters
        :return:
        """
        self.w_env = w_env
        self.model = PPO('MlpPolicy', self.w_env, verbose=1)
        self.model.load(path)

    def __repr__(self):
        return "PPO - MLP Policy"


