from agents.PPO_MLP_Agent import PpoMlp
from agents.A2C_MLP_Agent import A2cMlp
from agents.SAC_MLP_Agent import SacMlp
from agents.DQN_MLP_Agent import DqnMlp
from agents.AgentAbs import Agent
import gymnasium as gym
import datetime
import os
import shutil


class AgentsLoader:

    def __init__(self, w_env: gym.Env):
        self.w_env = w_env
        self.agents: [Agent] = []
        self.__load_agents()

    def __load_agents(self):
        self.agents.append(A2cMlp())
        self.agents.append(PpoMlp())
        self.agents.append(DqnMlp())
        # self.agents.append(SacMlp())

    def load_weights(self):
        for agent in self.agents:
            agent.load_model(w_env=self.w_env)

    def train(self, train_duration: int = 365 * 1000) -> float:
        """
        Train the agents and store the respective models.
        :param train_duration: how many episode run to find optimal policy and value function
        :return: duration is seconds
        """
        start = datetime.datetime.now()

        for agent in self.agents:
            self.w_env.reset()
            print(f"Training agent: {agent}")
            try:
                # Check if previous checkpoint exists...
                checkpoint_dir = f"models/checkpoints/"
                latest_model_path = None
                if os.path.exists(checkpoint_dir):
                    models = [f"models/checkpoints/{f}" for f in os.listdir(checkpoint_dir) if f.startswith(agent.id)]
                    if models:
                        latest_model_path = max(models, key=os.path.getctime)

                if latest_model_path:
                    agent.load_model(w_env=self.w_env, path=latest_model_path)
                    print(f"Loaded model from {latest_model_path}")
                # Train the agent
                agent.train(self.w_env, train_duration)
                agent.save_model()

                # Remove checkpoint if exists since i have full model saved
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)

            except KeyboardInterrupt:
                agent.save_model(path=f"models/checkpoints/{agent.id}_truncated")

        end = datetime.datetime.now()
        return (end-start).total_seconds()

    def load_weight(self, model_id: str = "") -> Agent:
        """
        Load weight for a single agent and return it.
        :param model_id: model to load weights for
        :return: agent loaded instance
        """
        for agent in self.agents:
            if agent.id == model_id:
                agent.load_model(w_env=self.w_env)
                return agent


