from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt


class RewardCallback(BaseCallback):
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
        for i, actions in enumerate(self.actions):
            if len(actions) == 2:
                ax2.plot(range(len(actions)), actions, label=f'Action {i + 1}')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.title(f'{self.description} Training Rewards and Actions over Time')
        plt.show()

