

class QuantityPrice(object):

    def __init__(self, quantity, price):
        self.quantity = quantity
        self.price = price
        return

    def get_quantity(self):
        return self.quantity

    def get_price(self):
        return self.price

    def get_value(self):
        return self.get_price() * self.get_quantity()

    def __repr__(self):
        return "%10.2f" % (self.get_value(), )