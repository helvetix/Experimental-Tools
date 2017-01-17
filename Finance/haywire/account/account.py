"""
Accounts have positions
"""
from position import Position


class Account(object):
    """An Account has a cash position, plus any number of other Positions.
    """

    CASH = "_Cash"

    def __init__(self, name, cash=0.0, positions={}):
        self.name = name
        self.cash = cash
        self.positions = positions
        return

    def withdraw_cash(self, amount):
        self.cash -= amount
        return self.cash

    def deposit_cash(self, amount):
        self.cash += amount
        return self.cash

    def _add_or_merge_position(self, position):
        if position.get_name() in self.positions:
            self.positions[position.get_name()].deposit(position)
        else:
            self.positions.update({position.get_name(): position})

        return self.positions[position.get_name()]

    def add_position(self, position):
        if self.cash >= position.get_value():
            self.cash -= position.get_value()
            self._add_or_merge_position(position)

        return

    def buy(self, name, quantity, last_price):
        required_cash = quantity * last_price

        assert self.cash >= required_cash

        if self.cash >= required_cash:
            new_position = Position(name, quantity, last_price)
            self._add_or_merge_position(new_position)

            position = self.positions[name]
            position.set_last_price(last_price)

            self.cash -= required_cash

        return

    def sell(self, name, quantity, last_price=None):
        position = self.positions[name]

        if last_price is not None:
            position.set_last_price(last_price)

        self.cash += position.sell(quantity)
        return

    def get_value(self):
        result = self.cash

        for position in self.positions.values():
            result += position.get_value()

        return result

    def __repr__(self):
        result = "Account %s\n" % (self.name)

        result += "  %s\n" % (Position.format(Account.CASH, None, None, self.cash),)

        for position in self.positions.values():
            result += "  %s\n" % position

        result += "================"
        result += "  %s\n" % (Position.format(None, None, None, account.get_value()),)
        return result

if __name__ == '__main__':
    account = Account("my-account", 10000.0)
    print account

    account.buy("ORCL", 100, 25.0)
    account.buy("ORCL", 100, 25.0)
    print account

    account.sell("ORCL", 100, 25.0)
    print account

    account.sell("ORCL", 100, 25.0)
    print account

    account.sell("ORCL", 100, 25.0)
    