

class Position(object):
    @staticmethod
    def format(name, quantity, last_price, total):
        quantity = quantity = "%10.2f" % (quantity,) if quantity is not None else "%10s" % ("",)
        last_price = last_price = "%10.2f" % (last_price,) if last_price is not None else "%10s" % ("",)
        total = total = "%10.2f" % (total,) if total is not None else "%10s" % ("",)
        name = name if name is not None else ""

        result = "%10s %s %s %s" % (name, quantity, last_price, total)
        return result

    def __init__(self, name, quantity=0.0, last_price=0.0):
        self.name = name
        self.last_price = last_price
        self.quantity = quantity
        self.history = []
        return

    def get_name(self):
        return self.name

    def get_last_price(self):
        return self.last_price

    def get_quantity(self):
        return self.quantity

    def get_history(self):
        return self.history

    def deposit(self, other_position):
        assert self.name == other_position.name
        self.last_price = other_position.last_price
        self.quantity += other_position.quantity
        self.history.append(other_position)
        return

    def withdraw(self, quantity):
        self.quantity -= quantity
        return

    def buy(self, quantity):
        value = quantity * self.last_price
        self.quantity += quantity
        return value

    def sell(self, quantity):
        assert quantity <= self.quantity

        value = quantity * self.last_price
        self.quantity -= quantity
        return value

    def set_last_price(self, last_price):
        self.last_price = last_price
        return self.get_value()

    def get_value(self):
        return self.last_price * self.quantity

    def __repr__(self):
        return Position.format(self.get_name(), self.get_quantity(), self.get_last_price(), self.get_value())
