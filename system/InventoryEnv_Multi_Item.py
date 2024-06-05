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
        self.state = self.warehouse.state
        self.action_space = spaces.MultiDiscrete([75*2, 76*2])
        self.step_duration = step_duration
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5 * len(self.warehouse.items),), dtype=np.float32)
        self.reward = 0
        self.end = self.warehouse.env.now

    def _get_observation(self):
        obs = []
        for item in self.warehouse.items:
            item_id = item.id
            state = self.warehouse.state[item_id]
            obs.extend((
                state.ip,
                state.qty_ordered_until_now,
                state.delta_time_last_order,
                state.orders_counter,
                state.order_rate
            ))
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=42, **kwargs):
        random.seed(seed)
        self.warehouse.env = simpy.Environment()
        self.warehouse.reset_system_attributes()
        self.warehouse.run_processes()
        return self._get_observation(), {}

    def step(self, actions, done_steps: int = 365*3):
        """
        :param actions: list of action to take
        :param done_steps: time to run before done for episode. Learn is mush bigger.
        :return:
        """
        info = {}
        done = False
        truncated = False
        for idx, action in enumerate(actions):
            item = self.warehouse.items[idx]
            info[f'stock_before_action_item_{item}'.replace(" ","")] = self.state[item.id].ip
            info[f'stock_after_action_item_{item}'.replace(" ","")] = self.state[item.id].ip + action
            info[f'qty_2_order_item_{item}'.replace(" ","")] = action
            self.warehouse.take_action(action, item)

        self.warehouse.env.run(until=self.end+self.step_duration)
        self.end = self.warehouse.env.now
        self.warehouse.update_costs()
        self.reward = -self.warehouse.day_total_cost[-1]
        return self._get_observation(), self.reward, False, False, info
