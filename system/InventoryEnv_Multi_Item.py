import random
from gymnasium.wrappers import NormalizeObservation
from agents.Wrapper import NormalizeRewardWrapper
from system.Inventory_Multi_Item import Warehouse
from gymnasium import spaces
import simpy
import numpy as np
import gymnasium as gym


class WarehouseEnv(gym.Env):
    def __init__(
        self,
        warehouse: Warehouse,
        step_duration: float
    ) -> None:
        super(WarehouseEnv, self).__init__()
        self.warehouse = warehouse
        self.action_space = spaces.Discrete(301)
        self.step_duration = step_duration
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5 * len(self.warehouse.items),), dtype=np.float32)
        self.reward = 0
        self.beginning = self.warehouse.env.now
        self.end = self.warehouse.env.now

    @classmethod
    def with_normalize_wrapper(cls, normalize=True, *args, **kwargs):
        env = cls(*args, **kwargs)
        if normalize:
            env = NormalizeObservation(env)
            env = NormalizeRewardWrapper(env)
        return env

    def _get_observation(self):
        obs = []
        for item in self.warehouse.items:
            item_id = item.id
            obs.extend((
                self.warehouse.inventory_levels[item_id],
                self.warehouse.items_ordered_currently[item_id],
                self.warehouse.delta_time_last_order[item_id],
                self.warehouse.orders_counter_currently[item_id],
                self.warehouse.order_rate(item)
            ))
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=42, **kwargs):
        random.seed(seed)
        self.warehouse.env = simpy.Environment()
        self.beginning = self.warehouse.env.now
        self.warehouse.reset_system_attributes()
        self.warehouse.run_processes()
        return self._get_observation(), {}

    def step(self, action: int, done_steps: int = 365*3, il_interval: [int] = [-5000, +5000]):
        """
        :param action: ty of item to order
        :param done_steps: time to run before done for episode. Learn is mush bigger.
        :param il_interval: interval level interval accepted before truncation
        :return:
        """
        info = {}
        idx = 1 if action >= 151 else 0
        action = action % 151
        item = self.warehouse.items[idx]
        self.warehouse.take_action(action, item)
        self.warehouse.env.run(until=self.warehouse.env.now+self.step_duration)
        reward = -self.warehouse.total_cost
        done = True if self.warehouse.env.now-self.beginning >= done_steps else False
        truncated = not (il_interval[0] <= self.warehouse.inventory_levels[item.id] <= il_interval[1])
        if truncated:
            remaining_time_steps = self.warehouse.env.now-self.beginning
            reward = self.truncated_cost(remaining_time_steps)
        return self._get_observation(), reward, done, truncated, info

    def truncated_cost(self, remaining_time: int, weight: int = 10):
        """

        :param remaining_time: time step remaining after truncation
        :param weight: weight to use to increment the cost of the truncation
        :return: weighted shortage cost considering time and proportional cost
        """
        # avg_fixed_policy -> -21000
        shortage_cost = sum(
            -min(self.warehouse.inventory_levels[item.id], 0) * self.warehouse.shortage_cost
            for item in self.warehouse.items
        )
        output = shortage_cost * remaining_time * weight
        return output
