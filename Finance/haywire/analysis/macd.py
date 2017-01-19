
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

    def macd(self):
        MACD_name = 'MACD' + self.parameters
        MACD_signal_name = 'MACDsignal' + self.parameters
        MACD_histogram_name = 'MACDhistogram' + self.parameters

        _fast = pd.Series.ewm(self.dataframe['Close'], span=self.fast, min_periods=self.slow-1).mean()
        _slow = pd.Series.ewm(self.dataframe['Close'], span=self.slow, min_periods=self.slow-1).mean()
        _macd = pd.Series(_fast - _slow, name=MACD_name)

        _signal = pd.Series(pd.Series.ewm(_macd, span=9, min_periods=8).mean(), name=MACD_signal_name)

        _histogram = pd.Series(_macd - _signal, name=MACD_histogram_name)

        self.macd_dataframe[MACD_name] = _macd
        self.macd_dataframe[MACD_signal_name] = _signal
        self.macd_dataframe[MACD_histogram_name] = _histogram

        self.trigger = _histogram[-1]

        return self.macd_dataframe

    def plot(self):
        self.macd_dataframe.plot(subplots=False, grid=True)
        return

    def update(self, value):
        print type(value)

        if self.dataframe is None:
            self.dataframe = pd.DataFrame([value])
        else:
            self.dataframe = self.dataframe.append([value])
        return

    def get_trigger(self):
        return self.trigger


if __name__ == "__main__":
    stock_dataframe = pd.DataFrame.from_csv('data/orcl-2015.csv')
    macd = MACD(fast=12, slow=26, signal=9)
    macd.update(stock_dataframe)
    macd_dataframe = macd.macd()

    print macd.get_trigger()

    print macd.macd_dataframe[-1:]

    stock_dataframe['Close'].plot()
    macd.plot()
#     plt.grid()
#     plt.show()
    macd2 = MACD(fast=12, slow=26, signal=9)
    for index, row in stock_dataframe.iterrows():
        macd2.update(row)
        macd2.macd()

    #macd2.update(stock_dataframe)
    macd2.macd()
    macd2.plot()
    plt.grid()
    plt.show()
   