from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


class RewardOldCallback(BaseCallback):
    def __init__(self, description, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.description = description
        self.rewards = []
        self.actions = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'])
        self.actions.append(self.locals['actions'])
        return True

    def plot_rewards(self):
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward', color=color)
        ax1.plot(self.rewards, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Action', color=color)
        ax2.plot(range(len(self.actions)), self.actions, color=color)

        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.title(f'{self.description} Training Rewards and Actions over Time')
        plt.legend(['Reward', 'Action'])
        plt.show()


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.actions = []
        self.progress_bar = None

    def _on_training_start(self) -> None:
        print("Training started...")
        self.progress_bar = tqdm(total=self.model._total_timesteps, desc="Training Progress")

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        # Increment progress bar at each step
        if self.progress_bar:
            self.progress_bar.update(1)
        # Track the reward and action at each timestep
        self.rewards.append(self.locals['rewards'][0])
        self.actions.append(self.locals['actions'][0])
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        print("Training completed!")
        if self.progress_bar:
            self.progress_bar.close()
        self.plot_rewards_and_actions()

    def plot_rewards_and_actions(self) -> None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        timesteps = np.arange(len(self.rewards))

        # Plot rewards
        ax1.plot(timesteps, self.rewards, color='b', label='Reward')
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Reward", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a second y-axis to plot the actions
        ax2 = ax1.twinx()
        ax2.plot(timesteps, self.actions, color='r', alpha=0.3, label='Actions')
        ax2.set_ylabel("Actions", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Calculate and display average reward and reward variance
        avg_reward = np.mean(self.rewards)
        reward_variance = np.var(self.rewards)
        description = f"Avg Reward: {avg_reward:.2f}, Reward Variance: {reward_variance:.2f}"
        plt.title(f"Training Rewards and Actions Taken\n{description}")

        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

