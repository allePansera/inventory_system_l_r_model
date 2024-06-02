class State:
    """
    Describes state about our POMDP problem.
    - Inventory position is current one
    - qty ordered until now is the qty that we are waiting to receive
    - delta time last order is delta time before last order
    - orders counter is the current number of order made
    - turnover rate is evaluated considering total sales and avg of inventory level
    - order rate is evaluated considering current exposed merch and current order exposed
    """
    def __init__(self,
                 inventory_position: int,
                 qty_ordered_until_now: int,
                 delta_time_last_order: float,
                 orders_counter: int,
                 order_rate: float) -> None:
        """

        :param inventory_position: qty in stocks considering current item
        :param qty_ordered_until_now: qty ordered considering current item
        :param delta_time_last_order: delta time with respect to last order execution
        :param orders_counter: total items ordered
        :param order_rate: order rate considering current exposed merch and current order exposed
        """
        self.ip = inventory_position
        self.qty_ordered_until_now = qty_ordered_until_now
        self.delta_time_last_order = delta_time_last_order
        self.orders_counter = orders_counter
        self.order_rate = order_rate

    def __str__(self):
        """
        Pretty prints the given State object
        """
        output = f"""Inventory Position: {self.ip} 
Delta Quantity Ordered Until Now: {self.qty_ordered_until_now}
Delta Time Last Order: {self.delta_time_last_order}
Delta Orders Counter: {self.orders_counter}
Order Rate: {self.order_rate}"""
        return output

    def __repr__(self):
        return self.__str__()
