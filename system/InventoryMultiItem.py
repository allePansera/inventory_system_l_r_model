from system.Inventory import Warehouse
from gymnasium import spaces
from collections import defaultdict
from typing import Callable, List
from system.State import State
from system.Item import Item
import random
import simpy
import statistics
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class WarehouseEnv(gym.Env):
    def __init__(
        self,
        warehouse: Warehouse,
        step_duration: float
    ) -> None:
        super(WarehouseEnv, self).__init__()
        self.warehouse = warehouse
        self.state = self.warehouse.state
        self.action_space = spaces.MultiDiscrete([1000] * len(self.warehouse.items))
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
            info[f'stock_before_action_item_{idx}'] = self.state[idx].ip
            info[f'stock_after_action_item_{idx}'] = self.state[idx].ip + action
            info[f'qty_2_order_item_{idx}'] = action

        self.warehouse.take_action(actions)
        self.warehouse.env.run(until=self.end+self.step_duration)
        self.end = self.warehouse.env.now
        self.warehouse.update_costs()
        self.reward = -1*self.warehouse.day_total_cost[-1]
        return self._get_observation(), self.reward, False, False, info


class Warehouse:
    def __init__(
        self,
        id: int,
        env: simpy.Environment,
        lead_time: Callable[[], float],
        demand_inter_arrival_mean_time: Callable[[], float],
        inventory_levels: List[Callable[[], float]],
        items: List[Item] = [],
        demand_distributions = [[[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]]],
        order_setup_cost: float = 32,
        order_incremental_cost: float = 3,
        holding_cost: float = 1,
        shortage_cost: float = 5,
        debug_mode: bool = False
    ) -> None:

        self.id = id
        self.env = env
        self.items = items
        self.demand_inter_arrival_mean_time = demand_inter_arrival_mean_time
        self.demand_distributions = demand_distributions
        self.init_inventory_levels = inventory_levels
        self.inventory_levels = [int(init_level()) for init_level in self.init_inventory_levels]
        self.lead_time = lead_time
        self.order_setup_cost = order_setup_cost
        self.order_incremental_cost = order_incremental_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.debug_mode = debug_mode

        # Attr. to handle inventory history
        self.inventory_history = [defaultdict(float) for _ in self.inventory_levels]
        self.it = [[(0, level)] for level in self.inventory_levels]
        self.last_inventory_levels = self.inventory_levels[:]
        self.last_inventory_level_timestamps = [0] * len(self.inventory_levels)
        self.total_sales = [0] * len(self.inventory_levels)

        # Attr. to evaluate system performance
        self.total_order_cost = 0
        self.daily_total_cost = []

        # Attr. used to implement state
        self.items_ordered_currently = [0] * len(self.inventory_levels)
        self.time_last_order = 0
        self.delta_time_last_order = 0.0
        self.orders_counter_currently = [0] * len(self.inventory_levels)

        self.state = [State(
            inventory_position=level,
            qty_ordered_until_now=0,
            delta_time_last_order=0,
            orders_counter=0,
            order_rate=0
        ) for level in self.inventory_levels]

        # Time unit
        self.day_duration = 1
        self.month_duration = 30 * self.day_duration

        # Handle all processes
        self.run_processes()

    def run_processes(self):
        self.env.process(self.demand_generator())

    def reset_system_attributes(self):
        self.inventory_levels = [int(init_level()) for init_level in self.init_inventory_levels]
        self.inventory_history = [defaultdict(float) for _ in self.inventory_levels]
        self.it = [[(0, level)] for level in self.inventory_levels]
        self.last_inventory_levels = self.inventory_levels[:]
        self.last_inventory_level_timestamps = [0] * len(self.inventory_levels)
        self.total_sales = [0] * len(self.inventory_levels)
        self.time_last_order = 0
        self.items_ordered_currently = [0] * len(self.inventory_levels)
        self.delta_time_last_order = 0.0
        self.orders_counter_currently = [0] * len(self.inventory_levels)
        self.daily_total_cost = []
        for state, level in zip(self.state, self.inventory_levels):
            state.ip = level
            state.delta_time_last_order = 0
            state.orders_counter = 0
            state.qty_ordered_until_now = 0
            state.order_rate = 0

    @property
    def total_items_ordered(self):
        return self.items_ordered_currently

    @property
    def turnover_rate(self):
        try:
            avg_inventory = [statistics.mean([el[1] for el in it]) for it in self.it]
            turnover_rates = [sales / avg if avg > 0 else 0 for sales, avg in zip(self.total_sales, avg_inventory)]
            return turnover_rates
        except ZeroDivisionError:
            return [0] * len(self.inventory_levels)
        except statistics.StatisticsError:
            return [0] * len(self.inventory_levels)

    @property
    def order_rate(self):
        try:
            order_rates = [items / orders if orders > 0 else 0 for items, orders in zip(self.items_ordered_currently, self.orders_counter_currently)]
            return order_rates
        except ZeroDivisionError:
            return [0] * len(self.inventory_levels)
        except statistics.StatisticsError:
            return [0] * len(self.inventory_levels)
        except RuntimeWarning:
            return [0] * len(self.inventory_levels)

    @property
    def total_holding_cost(self) -> float:
        return sum(
            sum(max(level, 0) * duration * self.holding_cost
            for level, duration in history.items())
            for history in self.inventory_history
        )

    @property
    def total_shortage_cost(self) -> float:
        return sum(
            sum(max(-level, 0) * duration * self.shortage_cost
            for level, duration in history.items())
            for history in self.inventory_history
        )

    @property
    def total_cost(self) -> float:
        return self.total_order_cost + self.total_holding_cost + self.total_shortage_cost

    @property
    def day_total_cost(self) -> [float]:
        return self.daily_total_cost

    def take_action(self, actions):
        for idx, action in enumerate(actions):
            if self.debug_mode: print(f"Ordered qty: {action} for item {idx} - {self.env.now}")
            self.env.process(self.order_given_qty(action, idx))
            if self.debug_mode: print(f"Merch received, current level: {self.inventory_levels[idx]} - {self.env.now}")

    def update_costs(self):
        if len(self.daily_total_cost) == 0:
            self.daily_total_cost.append(round(self.total_cost, 2))
        else:
            previous_cost = sum(self.daily_total_cost)
            self.daily_total_cost.append(round(self.total_cost-previous_cost, 2))

    def order_given_qty(self, qty_2_order: int, item_idx: int) -> None:
        self.orders_counter_currently[item_idx] += 1
        self.items_ordered_currently[item_idx] += qty_2_order
        self.total_sales[item_idx] += qty_2_order
        self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * qty_2_order)
        lead_time = self.lead_time()
        self.delta_time_last_order = self.env.now - self.time_last_order
        self.time_last_order = self.env.now

        self.state[item_idx].qty_ordered_until_now = self.items_ordered_currently[item_idx]
        self.state[item_idx].orders_counter = self.orders_counter_currently[item_idx]
        self.state[item_idx].turnover_rate = self.turnover_rate[item_idx]
        self.state[item_idx].order_rate = self.order_rate[item_idx]

        if self.debug_mode: print(f"DDT Printed from supplier for item {item_idx}...")
        if self.debug_mode: print(f"Estimated waiting time: {lead_time}")
        yield self.env.timeout(lead_time)
        if self.debug_mode: print(f"Dhl arrived for item {item_idx}...")
        self.inventory_levels[item_idx] += qty_2_order
        self.orders_counter_currently[item_idx] -= 1
        self.items_ordered_currently[item_idx] -= qty_2_order

        self.state[item_idx].qty_ordered_until_now = self.items_ordered_currently[item_idx]
        self.state[item_idx].orders_counter = self.orders_counter_currently[item_idx]
        self.state[item_idx].delta_time_last_order = self.delta_time_last_order
        self.state[item_idx].turnover_rate = self.turnover_rate[item_idx]
        self.state[item_idx].order_rate = self.order_rate[item_idx]

    def demand_generator(self) -> None:
        while True:
            demand_inter_arrival_time = self.demand_inter_arrival_mean_time()
            yield self.env.timeout(demand_inter_arrival_time)
            for idx, (pop, weights) in enumerate(self.demand_distributions):
                demand_size = random.choices(pop, weights=weights, k=1)[0]
                if self.debug_mode: print(f"Customer arrived and requires {demand_size} items of type {idx}")
                self.inventory_levels[idx] -= demand_size

    def system_description(self, output_path: str = "plot/"):
        fig, axs = plt.subplots(len(self.inventory_levels) + 1, 1, figsize=(12, 16))

        for idx in range(len(self.inventory_levels)):
            axs[idx].plot(
                [it[0] for it in self.it[idx]],
                [it[1] for it in self.it[idx]],
                color='blue',
                linewidth=2,
                label=f'Inventory Position {idx}'
            )
            axs[idx].fill_between(
                [it[0] for it in self.it[idx]],
                [it[1] for it in self.it[idx]],
                color='blue',
                alpha=0.2,
                label=f'Inventory Position {idx}'
            )
            axs[idx].set_title(f'I(t): Inventory Position over time for Item {idx}')
            axs[idx].set_xlabel('Simulation Time')
            axs[idx].set_ylabel('Inventory Position')
            axs[idx].legend()

        axs[-1].plot(
            range(len(self.daily_total_cost)),
            self.daily_total_cost,
            marker='o',
            color='red',
            linewidth=2,
            label='Daily Total Cost'
        )
        axs[-1].set_title('Daily Total Cost')
        axs[-1].set_xlabel('Day')
        axs[-1].set_ylabel('Cost (€)')
        axs[-1].legend()

        plt.tight_layout()
        plt.savefig(output_path)
        return self.__simulation_outcome()

    def __simulation_outcome(self):
        outcome = f"""
        Recap of inventory system current execution:
        Repetition: {self.id}
        ---
        €€:
        Total cost (€): {round(self.total_cost, 2)}
        Total holding cost (€): {round(self.total_holding_cost, 2)}
        Total shortage cost (€): {round(self.total_shortage_cost, 2)}
        ---
        """
        for idx in range(len(self.inventory_levels)):
            outcome += f"""
            Item {idx}:
            Final Inventory level: {self.inventory_levels[idx]}
            Total Items Ordered (delta): {self.items_ordered_currently[idx]}
            Total Orders (delta): {self.orders_counter_currently[idx]}
            Last time ordered (delta): {self.delta_time_last_order}
            Turnover rate: {round(self.turnover_rate[idx], 2)}
            Order rate: {round(self.order_rate[idx], 2)}
            """
        return outcome

