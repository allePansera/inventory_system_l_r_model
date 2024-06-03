from agents.DQN_MLP_Agent import DqnMlp
from agents.PPO_MLP_Agent import PpoMlp
from agents.A2C_MLP_Agent import A2cMlp
from agents.SAC_MLP_Agent import SacMlp
from agents.AgentAbs import Agent
import gym
import datetime


class AgentsLoader:

    def __init__(self, w_env: gym.Env):
        self.w_env = w_env
        self.agents: [Agent] = []
        self.__load_agents()

    def __load_agents(self):
        self.agents.append(PpoMlp())
        self.agents.append(DqnMlp())
        self.agents.append(A2cMlp())
        self.agents.append(SacMlp())

    def load_weights(self):
        for agent in self.agents:
            agent.load_model(w_env=self.w_env)

    def train(self, episode_duration: int = 1000, plot_rewards: bool = True) -> float:
        """
        Train the agents and store the respective models.
        :param episode_duration: how many episode run to find optimal policy and value function
        :param plot_rewards: choose whether plot or not reward progression during training
        :return: duration is seconds
        """
        start = datetime.datetime.now()

        for agent in self.agents:
            self.w_env.reset()
            print(f"Training agent: {agent}")
            agent.train(w_env=self.w_env, episode_duration=episode_duration, plot_rewards=plot_rewards)
            agent.save_model()

        end = datetime.datetime.now()
        return (end-start).total_seconds()



