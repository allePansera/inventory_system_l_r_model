import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable
from system.State import State
from system.Item import Item
import random
import simpy
import statistics


class Warehouse:
    """
    Model description:

    - lead time is calculated with a given distribution: tempo di consegna di un fornitore
    - demand distribution states different requests with random weight: quantità di materiale richiesta
    - inter arrival time is given from outside to assure sync from different simulation: tempo di arrivo tra una richiesta e un altra

    Pay attention that last time order is set to 0 at the beginning, maybe it's not a right assumption....

    Each month we check for inventory level and then ask the agent what to do

    State is updated within 2 events:
    - order executed (after receiving goods)
    - i.p. variation (inside setter method :) )
    """

    def __init__(
            self,
            id: int,
            env: simpy.Environment,
            lead_time: Callable[[], float],
            demand_inter_arrival_mean_time: Callable[[], float],
            inventory_level: Callable[[], float],
            items: [Item] = [],
            demand_distribution = [[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]],
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
        self.demand_distribution = demand_distribution
        self.init_inventory_level = inventory_level
        self._inventory_level = int(self.init_inventory_level())
        self.lead_time = lead_time
        self.order_setup_cost = order_setup_cost
        self.order_incremental_cost = order_incremental_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.debug_mode = debug_mode

        # Attr. to handle inventory history
        self.inventory_history: dict[float, float] = defaultdict(float)
        self.it: list[tuple[float, float]] = [(0, self._inventory_level)]
        self.last_inventory_level: float = self._inventory_level
        self.last_inventory_level_timestamp: float = 0
        self.total_sales: int = 0

        # Attr. to evaluate system performance
        self.total_order_cost = 0
        self.daily_total_cost = []

        # Attr. used to implement state
        self.items_ordered_currently = 0
        self.time_last_order = 0
        self.delta_time_last_order = 0.0
        self.orders_counter_currently = 0

        self.state = State(
            inventory_position=self._inventory_level,
            qty_ordered_until_now=self.items_ordered_currently,
            delta_time_last_order=self.delta_time_last_order,
            orders_counter=self.orders_counter_currently,
            order_rate=0
        )

        # Time unit
        self.day_duration = 1
        self.month_duration = 30 * self.day_duration

        # Handle all processes
        self.run_processes()

    def run_processes(self):
        """
        Method practical to re_execute processes while re-declaring simpy.Environment
        :return:
        """
        # Execute system tasks
        self.env.process(self.demand_generator())

    def reset_system_attributes(self):
        """
        Method useful for gym.Env reset()
        :return:
        """
        self._inventory_level = int(self.init_inventory_level())
        self.inventory_history: dict[float, float] = defaultdict(float)
        self.it: list[tuple[float, float]] = [(0, self._inventory_level)]
        self.last_inventory_level: float = self._inventory_level
        self.last_inventory_level_timestamp: float = 0
        self.total_sales = 0
        self.time_last_order = 0
        self.items_ordered_currently = 0
        self.delta_time_last_order = 0.0
        self.orders_counter_currently = 0

        # Attr. to evaluate system performance
        self.daily_total_cost = []
        self.state.ip = self._inventory_level
        self.state.delta_time_last_order = 0
        self.state.orders_counter = 0
        self.state.qty_ordered_until_now = 0
        self.state.order_rate = 0

    @property
    def inventory_level(self):
        return self._inventory_level

    @property
    def total_items_ordered(self):
        return self.items_ordered_currently

    @property
    def turnover_rate(self):
        try:
            inventory_position_history_list = [el[1] for el in self.it]
            avg_inventory = statistics.mean(inventory_position_history_list)
            if avg_inventory == 0:
                return 0
            return self.total_sales/avg_inventory
        except ZeroDivisionError:
            return 0
        except statistics.StatisticsError:
            return 0

    @property
    def order_rate(self):
        try:
            if self.orders_counter_currently == 0:
                return 0
            return self.items_ordered_currently / self.orders_counter_currently
        except ZeroDivisionError:
            return 0
        except statistics.StatisticsError:
            return 0
        except RuntimeWarning:
            return 0

    @inventory_level.setter
    def inventory_level(self, value):
        """
        Setter attr. to update all cost when updating the inventory level
        :param value: new value to set
        :return:
        """
        # Update last inventory level duration
        self.inventory_history[self.last_inventory_level] += self.env.now - self.last_inventory_level_timestamp

        # Set new inventory level
        self._inventory_level = value
        self.it.append((self.env.now, self._inventory_level))

        # Update state property of IP
        self.state.ip = self._inventory_level

        # Update last inventory level
        self.last_inventory_level_timestamp = self.env.now
        self.last_inventory_level = self._inventory_level

    @property
    def total_holding_cost(self) -> float:
        return sum(
            max(inventory_level, 0) * duration * self.holding_cost
            for inventory_level, duration in self.inventory_history.items()
        )

    @property
    def total_shortage_cost(self) -> float:
        return sum(
            max(-inventory_level, 0) * duration * self.shortage_cost
            for inventory_level, duration in self.inventory_history.items()
        )

    @property
    def total_cost(self) -> float:
        return self.total_order_cost + self.total_holding_cost + self.total_shortage_cost

    @property
    def day_total_cost(self) -> [float]:
        return self.daily_total_cost

    def take_action(self, action):
        """
        Used to perform the order
        :param action: qty. to order, our model's action
        :return:
        """
        if self.debug_mode: print(f"Ordered qty: {action} - {self.env.now}")
        self.env.process(self.order_given_qty(action))
        if self.debug_mode: print(f"Merch received, current level: {self.inventory_level} - {self.env.now}")

    def update_costs(self):
        """
        Method used to update system's cost
        :return:
        """
        # Daily cost check
        if len(self.daily_total_cost) == 0:
            self.daily_total_cost.append(round(self.total_cost, 2))
        else:
            previous_cost = sum(self.daily_total_cost)
            self.daily_total_cost.append(round(self.total_cost-previous_cost,2))

    def order_given_qty(self, qty_2_order: int) -> None:
        """
        Order specific qty.
        :param qty_2_order: qty. to oder
        :return: nothing
        """
        self.orders_counter_currently += 1
        self.items_ordered_currently += qty_2_order
        self.total_sales += qty_2_order
        self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * qty_2_order)
        lead_time = self.lead_time()
        self.delta_time_last_order = self.env.now - self.time_last_order
        self.time_last_order = self.env.now
        # Update last order info before receiving merch
        self.state.qty_ordered_until_now = self.items_ordered_currently
        self.state.orders_counter = self.orders_counter_currently
        self.state.turnover_rate = self.turnover_rate
        self.state.order_rate = self.order_rate
        if self.debug_mode: print(f"DDT Printed from supplier...")
        if self.debug_mode: print(f"Estimated waiting time: {lead_time}")
        yield self.env.timeout(lead_time)
        if self.debug_mode: print(f"Dhl arrived...")
        self.inventory_level += qty_2_order
        # Remove currently exposed order and qty
        self.orders_counter_currently -= 1
        self.items_ordered_currently -= qty_2_order
        # Update state property of orders made until now
        self.state.qty_ordered_until_now = self.items_ordered_currently
        self.state.orders_counter = self.orders_counter_currently
        self.state.delta_time_last_order = self.delta_time_last_order
        self.state.turnover_rate = self.turnover_rate
        self.state.order_rate = self.order_rate

    def demand_generator(self) -> None:
        """
        Simulate process of item demand from a randomly arrived customer.
        Inter-arrival time is modelled with expovariate distribution.
        :return:
        """
        while True:
            demand_inter_arrival_time = self.demand_inter_arrival_mean_time()
            pop, weights = self.demand_distribution
            demand_size = random.choices(pop, weights=weights, k=1)[0]
            yield self.env.timeout(demand_inter_arrival_time)
            if self.debug_mode: print(f"Customer arrived and requires {demand_size} items")
            self.inventory_level -= demand_size

    def system_description(self, output_path: str = "plot/"):
        """
        Plot the inventory level over time and the daily total cost.
        :param output_path: path to use for storing plot figure
        :return: textual description of system
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 0.2]})

        # Plot inventory level over time
        ax1.plot(
            [it[0] for it in self.it],
            [it[1] for it in self.it],
            color='blue',
            linewidth=2,
            label='Inventory Position'
        )
        ax1.fill_between(
            [it[0] for it in self.it],
            [it[1] for it in self.it],
            color='blue',
            alpha=0.2,
            label='Inventory Position'
        )
        ax1.set_title('I(t): Inventory Position over time')
        ax1.set_xlabel('Simulation Time')
        ax1.set_ylabel('Inventory Position')
        ax1.legend()

        # Plot daily total cost
        ax2.plot(
            range(len(self.daily_total_cost)),
            self.daily_total_cost,
            marker='o',
            color='red',
            linewidth=2,
            label='Daily Total Cost'
        )
        ax2.set_title('Daily Total Cost')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cost (€)')
        ax2.legend()

        # Add simulation outcome to the plot
        textual_outcome = self.__simulation_outcome()
        ax3.text(
            0.5,
            0.5,
            textual_outcome,
            ha='center',
            va='center',
            transform=ax3.transAxes,
            color='black',
            fontsize=12
        )
        ax3.axis('off')  # Hide the axes for the text plot

        plt.tight_layout()
        plt.savefig(output_path)
        return textual_outcome

    def __simulation_outcome(self):
        return f"""
        
        Recap of inventory system current execution:
        Repetition: {self.id}
        ---
        €€:\n
        Total cost (€): {round(self.total_cost,2)}
        Total holding cost (€): {round(self.total_holding_cost,2)}
        Total shortage cost (€): {round(self.total_shortage_cost,2)}
        ---\n
        Final Inventory level: {self.inventory_level}
        Total Items Ordered (delta): {self.total_items_ordered}
        Total Orders (delta): {self.orders_counter_currently}
        Last time ordered (delta): {self.delta_time_last_order}
        Turnover rate: {round(self.turnover_rate,2)}
        Order rate: {round(self.order_rate,2)}
        """








