from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def train(self, env, epochs, plot_reward):
        pass

    @abstractmethod
    def predict(self, obs):
        pass
    @abstractmethod
    def load_model(self, env, filepath):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass

