import random

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
        self.warehouse.reset_system_attributes()
        self.warehouse.run_processes()
        return self._get_observation(), {}

    def step(self, action: int, done_steps: int = 365*3, r_interval: [int] = [-500, +500]):
        """
        :param action: ty of item to order
        :param done_steps: time to run before done for episode. Learn is mush bigger.
        :param r_interval: reward interval accepted before truncation
        :return:
        """
        info = {}
        idx = 1 if action >= 151 else 0
        action = action % 151
        item = self.warehouse.items[idx]
        self.warehouse.take_action(action, item)
        self.warehouse.env.run(until=self.end+self.step_duration)
        self.end = self.warehouse.env.now
        self.reward = -self.warehouse.total_cost
        done = True if self.warehouse.env.now-self.beginning >= done_steps else False
        truncated = True if r_interval[0] <= self.reward <= r_interval[1] else False
        return self._get_observation(), self.reward, done, truncated, info
