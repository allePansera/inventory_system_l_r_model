from collections import defaultdict
from typing import Callable, List
from system.State import State
from system.Item import Item
import random
import simpy
import statistics
import matplotlib.pyplot as plt


class Warehouse:
    def __init__(
        self,
        id: int,
        env: simpy.Environment,
        inventory_levels: List[Callable[[], float]],
        items: List[Item] = [],
        order_setup_cost: float = 10,
        order_incremental_cost: float = 3,
        holding_cost: float = 1,
        shortage_cost: float = 7,
        debug_mode: bool = False
    ) -> None:

        self.id = id
        self.env = env
        self.items = items
        self.init_inventory_levels = inventory_levels
        self.inventory_levels = {
                                item.id: int(init_level())
                                for item, init_level
                                in zip(self.items, self.init_inventory_levels)
                            }
        self.order_setup_cost = order_setup_cost
        self.order_incremental_cost = order_incremental_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.debug_mode = debug_mode

        # Attr. to handle inventory history
        self.inventory_history = {item.id: {} for item in self.items}
        self.it = {item.id: [(0, self.inventory_levels[item.id])] for item in self.items}
        self.last_inventory_levels = self.inventory_levels
        self.last_inventory_level_timestamps = {item.id: 0 for item in self.items}
        self.total_sales = {item.id: 0 for item in self.items}

        # Attr. to evaluate system performance
        self.total_order_cost_per_item = {item.id: 0 for item in self.items}
        self.daily_total_cost = []

        # Attr. used to implement state
        self.items_ordered_currently = {item.id: 0 for item in self.items}
        self.time_last_order = {item.id: 0 for item in self.items}
        self.delta_time_last_order = {item.id: 0 for item in self.items}
        self.orders_counter_currently = {item.id: 0 for item in self.items}

        self.state = {item.id: State(
            inventory_position=self.inventory_levels[item.id],
            qty_ordered_until_now=0,
            delta_time_last_order=0,
            orders_counter=0,
            order_rate=0
        ) for item in self.items}

        # Time unit
        self.day_duration = 1
        self.month_duration = 30 * self.day_duration

        # Handle all item processes
        self.run_processes()

    def run_processes(self):
        self.env.process(self.update_costs())
        for item in self.items:
            self.env.process(self.demand_generator(item))
            self.env.process(self.order_qty(item))

    def reset_system_attributes(self):
        # Attr. to handle inventory levels
        self.inventory_levels = {
            item.id: int(init_level())
            for item, init_level
            in zip(self.items, self.init_inventory_levels)
        }
        # Attr. to handle inventory history
        self.inventory_history = {item.id: {} for item in self.items}
        self.it = {item.id: [(0, self.inventory_levels[item.id])] for item in self.items}
        self.last_inventory_levels = self.inventory_levels
        self.last_inventory_level_timestamps = {item.id: 0 for item in self.items}
        self.total_sales = {item.id: 0 for item in self.items}
        # Attr. to evaluate system performance
        self.total_order_cost_per_item = {item.id: 0 for item in self.items}
        self.daily_total_cost = []
        # Attr. used to implement state
        self.items_ordered_currently = {item.id: 0 for item in self.items}
        self.time_last_order = {item.id: 0 for item in self.items}
        self.delta_time_last_order = {item.id: 0 for item in self.items}
        self.orders_counter_currently = {item.id: 0 for item in self.items}
        for item, level in zip(self.items, self.init_inventory_levels):
            self.state[item.id].ip = level()
            self.state[item.id].delta_time_last_order = 0
            self.state[item.id].orders_counter = 0
            self.state[item.id].qty_ordered_until_now = 0
            self.state[item.id].order_rate = 0

    @property
    def total_items_ordered(self):
        return sum(self.items_ordered_currently[item.id] for item in self.items)

    def turnover_rate(self, item: Item):
        try:
            avg_inventory = statistics.mean([el[1] for el in self.it[item.id]])
            turnover_rates = self.total_sales[item.id] / avg_inventory
            return turnover_rates
        except ZeroDivisionError:
            return 0
        except statistics.StatisticsError:
            return 0

    def order_rate(self, item: Item):
        try:
            order_rates = self.items_ordered_currently[item.id]/self.orders_counter_currently[item.id]
            return order_rates
        except ZeroDivisionError:
            return 0
        except statistics.StatisticsError:
            return 0
        except RuntimeWarning:
            return 0

    @property
    def total_holding_cost(self) -> float:
        return sum(
            sum(max(level, 0) * duration * self.holding_cost
            for level, duration in self.inventory_history[item.id].items())
            for item in self.items
        )

    @property
    def total_shortage_cost(self) -> float:
        return sum(
            sum(max(-level, 0) * duration * self.shortage_cost
            for level, duration in self.inventory_history[item.id].items())
            for item in self.items
        )

    @property
    def total_order_cost(self):
        return sum(self.total_order_cost_per_item[item.id] for item in self.items)

    @property
    def total_cost(self) -> float:
        return self.total_order_cost + self.total_holding_cost + self.total_shortage_cost

    @property
    def day_total_cost(self) -> [float]:
        return self.daily_total_cost

    def update_costs(self):
        """
        Called by Gym to update costs at each episode
        :return:
        """
        while True:
            yield self.env.timeout(self.day_duration)
            if len(self.daily_total_cost) == 0:
                self.daily_total_cost.append(round(self.total_cost, 2))
            else:
                previous_cost = sum(self.daily_total_cost)
                self.daily_total_cost.append(round(self.total_cost-previous_cost, 2))

    def order_qty(self, item: Item) -> None:
        """
        Per Item order a specific qty when items are under specific shoulder.
        :param item: item considered
        :return:
        """
        while True:
            if self.inventory_levels[item.id] < item.s_min:
                qty_2_order = item.s_max - self.inventory_levels[item.id]
                if self.debug_mode: print(f"{item} is under s_min, make an order of {qty_2_order} units")
                self.orders_counter_currently[item.id] += 1
                self.items_ordered_currently[item.id] += qty_2_order
                self.total_sales[item.id] += qty_2_order
                self.total_order_cost_per_item[item.id] += (self.order_setup_cost + self.order_incremental_cost * qty_2_order)
                lead_time = item.lead_time()
                self.delta_time_last_order[item.id] = self.env.now - self.time_last_order[item.id]
                self.time_last_order[item.id] = self.env.now

                self.state[item.id].qty_ordered_until_now = self.items_ordered_currently[item.id]
                self.state[item.id].orders_counter = self.orders_counter_currently[item.id]
                self.state[item.id].order_rate = self.order_rate(item)

                if self.debug_mode: print(f"DDT Printed from supplier for item {item}...")
                if self.debug_mode: print(f"Estimated waiting time: {lead_time}")
                yield self.env.timeout(lead_time)
                if self.debug_mode: print(f"Dhl arrived for item {item}...")
                self.inventory_levels[item.id] += qty_2_order
                self.orders_counter_currently[item.id] -= 1
                self.items_ordered_currently[item.id] -= qty_2_order

                self.state[item.id].qty_ordered_until_now = self.items_ordered_currently[item.id]
                self.state[item.id].orders_counter = self.orders_counter_currently[item.id]
                self.state[item.id].delta_time_last_order = self.delta_time_last_order[item.id]
                self.state[item.id].order_rate = self.order_rate(item)

    def demand_generator(self, item: Item) -> None:
        """
        Simulate the requests from customer of each item.
        It's supposed that if one customer comes then we make an order per each item.
        :param item: Item instance to consider to simulate generator
        :return:
        """
        while True:
            demand_inter_arrival_time = item.demand_inter_arrival_time()
            yield self.env.timeout(demand_inter_arrival_time)
            pop, weights = item.demand_distribution
            demand_size = random.choices(pop, weights=weights, k=1)[0]
            if self.debug_mode: print(f"Customer arrived and requires {demand_size} items of type {item}")
            self.inventory_levels[item.id] -= demand_size

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
        for item in self.items:
            outcome += f"""
            Item {item}:
            Final Inventory level: {self.inventory_levels[item.id]}
            Total Items Ordered (delta): {self.items_ordered_currently[item.id]}
            Total Orders (delta): {self.orders_counter_currently[item.id]}
            Last time ordered (delta): {self.delta_time_last_order[item.id]}
            Turnover rate: {round(self.turnover_rate(item), 2)}
            Order rate: {round(self.order_rate(item), 2)}
            """
        return outcome
