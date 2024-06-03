from typing import Callable, List


class Item:

    def __init__(self,
                 id: str,
                 description: str,
                 lead_time: Callable[[], float],
                 demand_inter_arrival_time: Callable[[], float],
                 demand_distribution: [],
                 s_min: int = 20,
                 s_max: int = 60):
        """

        :param id: item's ID
        :param description: item's description
        :param lead_time: lead time of the supplier
        :param demand_inter_arrival_time: demand inter-arrival time of the item
        :param demand_distribution: demand distribution of the item
        :param s_min: minimum stock level
        :param s_max: maximum stock level
        """
        self.id = id
        self.description = description
        self.lead_time = lead_time
        self.demand_inter_arrival_time = demand_inter_arrival_time
        self.demand_distribution = demand_distribution
        self.s_min = s_min
        self.s_max = s_max

    def __repr__(self):
        return f"({self.id}){self.description}"
