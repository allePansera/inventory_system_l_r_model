from typing import Callable, List
from system.Item import Item
import random
import simpy
import statistics


class Warehouse:
    def __init__(
            self,
            id: int,
            env: simpy.Environment,
            inventory_levels: List[Callable[[], float]],
            items: List[Item] = (),
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

        # Define sys. attr
        self.total_order_cost_per_item = {}
        # Attr. used to define system state
        self.items_ordered_currently = {}
        self.time_last_order = {}
        self.delta_time_last_order = {}
        self.orders_counter_currently = {}
        self.reset_system_attributes()

        # Time unit
        self.day_duration = 1
        self.month_duration = 30 * self.day_duration

        # Costs History
        self.daily_total_cost = [self.total_cost]
        self.yesterday_cost = self.total_cost

        # Handle all item processes
        self.run_processes()

    def run_processes(self):
        for item in self.items:
            pid = self.env.process(self.demand_generator(item))
            self.process_list.append(pid)
        pid = self.env.process(self.update_cost_history())
        self.process_list.append(pid)

    def reset_system_attributes(self):
        # Attr. to handle inventory levels
        self.inventory_levels = {
            item.id: int(init_level())
            for item, init_level
            in zip(self.items, self.init_inventory_levels)
        }

        # Attr. to evaluate system performance
        self.total_order_cost_per_item = {item.id: 0 for item in self.items}
        # Attr. used to define system state
        self.items_ordered_currently = self.total_order_cost_per_item.copy()
        self.time_last_order = self.total_order_cost_per_item.copy()
        self.delta_time_last_order = self.total_order_cost_per_item.copy()
        self.orders_counter_currently = self.total_order_cost_per_item.copy()

        if not hasattr(self, 'process_list'): self.process_list = []

        # Clean all processes
        for pid in self.process_list:
            pid.interrupt()
            self.process_list.remove(pid)

    def order_rate(self, item: Item):
        try:
            order_rates = self.items_ordered_currently[item.id] / self.orders_counter_currently[item.id]
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
            max(self.inventory_levels[item.id], 0) * self.holding_cost
            for item in self.items
        )

    @property
    def total_shortage_cost(self) -> float:
        return sum(
            -min(self.inventory_levels[item.id], 0) * self.shortage_cost
            for item in self.items
        )

    @property
    def total_order_cost(self):
        return sum(self.total_order_cost_per_item[item.id] for item in self.items)

    @property
    def total_cost(self) -> float:
        return self.total_order_cost + self.total_holding_cost + self.total_shortage_cost

    def take_action(self, action: int, item: Item):
        """
        Perform an order given an item.
        :param action: qty to order per each item
        :param item: item to consider for performed action
        :return:
        """
        if self.debug_mode: print(f"Ordered qty: {action} for item {item} - {self.env.now}")
        self.env.process(self.order_given_qty(action, item))
        if self.debug_mode: print(f"Merch received, current level: {self.inventory_levels[item]} - {self.env.now}")

    def order_given_qty(self, qty_2_order: int, item: Item) -> None:
        """
        Per Item order a given qty
        :param qty_2_order: quantity to order
        :param item: item considered
        :return:
        """
        self.orders_counter_currently[item.id] += 1
        self.items_ordered_currently[item.id] += qty_2_order
        self.total_order_cost_per_item[item.id] += (self.order_setup_cost + self.order_incremental_cost * qty_2_order)
        lead_time = item.lead_time()
        self.delta_time_last_order[item.id] = self.env.now - self.time_last_order[item.id]
        self.time_last_order[item.id] = self.env.now
        if self.debug_mode: print(f"DDT Printed from supplier for item {item}...")
        if self.debug_mode: print(f"Estimated waiting time: {lead_time}")
        yield self.env.timeout(lead_time)
        if self.debug_mode: print(f"DHL arrived for item {item}...")
        self.inventory_levels[item.id] += qty_2_order
        self.orders_counter_currently[item.id] -= 1
        self.items_ordered_currently[item.id] -= qty_2_order

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

    def update_cost_history(self):
        """
        Update cost history list with the total cost of the system.
        """
        while True:
            yield self.env.timeout(self.day_duration)
            total_cost = self.total_cost
            self.daily_total_cost.append(round(total_cost - self.yesterday_cost, 2))
            self.yesterday_cost = total_cost
