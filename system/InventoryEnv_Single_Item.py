from .Inventory import Warehouse
import gymnasium as gym
from gymnasium import spaces
import simpy
import numpy as np


class WarehouseEnv(gym.Env):
    """
    Model description:

    check for Inventory.py

    - Adapted to reinforcement learning model.
    - Action can be a number from 0 to 100.
    - Reward is related to current month's cost.
    - This is a non-terminal system.
    """

    def __init__(
            self,
            warehouse: Warehouse,
            step_duration: float
    ) -> None:
        """
        :param warehouse: Warehouse instance
        :param step_duration: how many month the step should last
        :param step_duration: how full episodes lasts
        """
        super(WarehouseEnv, self).__init__()
        self.warehouse = warehouse
        # Save environment state
        self.state = self.warehouse.state
        # Define actions
        self.action_space = spaces.Discrete(1000)
        # Set step duration (month)
        self.step_duration = step_duration
        # Define possible set of observation
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        # Define reward class attribute
        self.reward = 0
        # End
        self.end = self.warehouse.env.now

    def _get_observation(self):
        return np.array([
            self.warehouse.state.ip,
            self.warehouse.state.qty_ordered_until_now,
            self.warehouse.state.delta_time_last_order,
            self.warehouse.state.orders_counter,
            self.warehouse.state.order_rate
        ],
        dtype=np.float32)

    def reset(self, **kwargs):
        """
        Reset state of my warehouse
        :return: system's state
        """
        # Revert simpy environment
        self.warehouse.env = simpy.Environment()
        # Clean previous attributes
        self.warehouse.reset_system_attributes()
        # Re_execute all process
        self.warehouse.run_processes()
        # Return current state
        return self._get_observation(), {}

    def step(self, action):
        """
        Analyze a system until a step is done. A step is a month inside system
        :param action: number from 0 to 100 regarding qty. that we can order
        :return: system's state, reward, done, truncated, info
        """
        # Info attr. is explainatory
        info = {
            'stock_before_action': self.state.ip,
            'stock_after_action': self.state.ip + action,
            'qty_2_order': action,
        }
        # Perform the action
        self.warehouse.take_action(action)
        # Run system for 1 step
        self.warehouse.env.run(until=self.end+self.step_duration)
        self.end = self.warehouse.env.now
        # Update costs array
        self.warehouse.update_costs()
        # Total cost over last day is attribute reward, negative since we maximize it
        self.reward = -1*self.warehouse.day_total_cost[-1]
        return self._get_observation(), self.reward, False, False, info







