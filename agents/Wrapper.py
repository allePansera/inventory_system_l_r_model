import gymnasium as gym
import numpy as np


class NormalizeRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, gamma=0.99):
        super(NormalizeRewardWrapper, self).__init__(env)
        self.rewards = []
        self.gamma = gamma
        self.mean = 0.0
        self.std = 1.0

    def reward(self, reward):
        self.rewards.append(reward)
        discounted_rewards = self._discounted_rewards(self.rewards, self.gamma)
        self.mean = np.mean(discounted_rewards)
        self.std = np.std(discounted_rewards) + 1e-8  # Evita divisioni per zero
        normalized_reward = (reward - self.mean) / self.std
        return normalized_reward

    def _discounted_rewards(self, rewards, gamma):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted[t] = running_add
        return discounted


