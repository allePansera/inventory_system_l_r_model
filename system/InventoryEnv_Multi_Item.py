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
        self.action_space = spaces.MultiDiscrete([75*2, 76.125*2])
        self.step_duration = step_duration
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5 * len(self.warehouse.items),), dtype=np.float32)
        self.reward = 0
        self.end = self.warehouse.env.now

    def _get_observation(self):
        obs = []
        for state in self.warehouse.state:
            obs.extend([
                state.ip,
                state.qty_ordered_until_now,
                state.delta_time_last_order,
                state.orders_counter,
                state.order_rate
            ])
        return np.array(obs, dtype=np.float32)

    def reset(self, **kwargs):
        self.warehouse.env = simpy.Environment()
        self.warehouse.reset_system_attributes()
        self.warehouse.run_processes()
        return self._get_observation(), {}

    def step(self, actions):
        info = {}
        for idx, action in enumerate(actions):
            item = self.warehouse.items[idx]
            info[f'stock_before_action_item_{item}'] = self.state[item].ip
            info[f'stock_after_action_item_{item}'] = self.state[item].ip + action
            info[f'qty_2_order_item_{item}'] = action
            self.warehouse.take_action(action, item)

        self.warehouse.env.run(until=self.end+self.step_duration)
        self.end = self.warehouse.env.now
        self.warehouse.update_costs()
        self.reward = -1*self.warehouse.day_total_cost[-1]
        return self._get_observation(), self.reward, False, False, info
