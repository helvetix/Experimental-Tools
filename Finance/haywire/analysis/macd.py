
import pandas as pd
import matplotlib.pyplot as plt
from trading_indicator import TradingIndicator


class MACD(TradingIndicator):
    """
    """

    def __init__(self, fast=12, slow=26, signal=9, dataframe=None):
        super(MACD, self).__init__("MACD")

        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.parameters = '(%d,%d,%d)' % (self.fast, self.slow, self.signal)
        self.dataframe = dataframe
        self.macd_dataframe = pd.DataFrame()

        return

    def old_MACD(self, df, n_fast, n_slow):
        EMAfast = pd.Series.ewm(df['Close'], ignore_na=False, span=n_fast, min_periods=n_slow - 1, adjust=True).mean()
        EMAslow = pd.Series.ewm(df['Close'], ignore_na=False, span=n_slow, min_periods=n_slow - 1, adjust=True).mean()
        MACD = 'MACD_' + str(n_fast) + '_' + str(n_slow)
        df[MACD] = pd.Series(EMAfast - EMAslow, name=MACD)

        MACDsignal = pd.Series(pd.Series.ewm(df[MACD], ignore_na=False, span=9, min_periods=8, adjust=True).mean(),
                               name='MACDsignal_' + str(n_fast) + '_' + str(n_slow))

        MACDindicator = pd.Series(df[MACD] - MACDsignal, name='MACDhistogram_' + str(n_fast) + '_' + str(n_slow))

        df = df.join(MACDsignal)
        df = df.join(MACDindicator)
        return df

    def macd(self, stock_dataframe):
        MACD_name = 'MACD' + self.parameters
        MACD_signal_name = 'MACDsignal' + self.parameters
        MACD_histogram_name = 'MACDhistogram' + self.parameters

        EMAfast = pd.Series.ewm(self.dataframe['Close'], span=self.fast, min_periods=self.slow-1).mean()
        EMAslow = pd.Series.ewm(self.dataframe['Close'], span=self.slow, min_periods=self.slow-1).mean()
        _macd = pd.Series(EMAfast - EMAslow, name=MACD_name)

        _signal = pd.Series(pd.Series.ewm(_macd, span=9, min_periods=8).mean(), name=MACD_signal_name)

        _histogram = pd.Series(_macd - _signal, name=MACD_histogram_name)

        self.macd_dataframe[MACD_name] = _macd
        self.macd_dataframe[MACD_signal_name] = _signal
        self.macd_dataframe[MACD_histogram_name] = _histogram

        self.trigger = _histogram[-1]

        stock_dataframe[MACD_name] = pd.Series(EMAfast - EMAslow, name=MACD_name)
        result = stock_dataframe.join(_signal)
        result = result.join(_histogram)

        return result

    def plot(self):
        self.macd_dataframe.plot(subplots=False, grid=True)
        return

    def update(self, open_high_low_close_volume):
        return

    def get_trigger(self):
        return self.trigger


if __name__ == "__main__":
    stock_dataframe = pd.DataFrame.from_csv('data/orcl-2015.csv')
    macd = MACD(fast=12, slow=26, signal=9, dataframe=stock_dataframe)
    macd_dataframe = macd.macd(stock_dataframe)

    print macd.get_trigger()

    print macd.macd_dataframe[-1:]

    stock_dataframe['Close'].plot()
    macd.plot()
    plt.grid()
    plt.show()
