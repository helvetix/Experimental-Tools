import abc


class TradingIndicator(object):

    def __init__(self, name):
        self.name = name
        self.trigger = None
        return

    @abc.abstractmethod
    def update(self, open_high_low_close_volume):
        return

    @abc.abstractmethod
    def get_trigger(self):
        """Get the value of the trading trigger.
        """

        return self.trigger

    def get_name(self):
        return self.name
