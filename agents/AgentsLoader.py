from agents.PPO_MLP_Agent import PpoMlp
from agents.A2C_MLP_Agent import A2cMlp
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
        self.agents.append(PpoMlp())
        self.agents.append(A2cMlp())

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
                agent.train(w_env=self.w_env, episode_duration=episode_duration, plot_rewards=plot_rewards)
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


