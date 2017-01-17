#!/usr/local/bin/python2.7
# encoding: utf-8
'''
stock.macd -- shortdesc

'''

import sys
import os

import portfolio as Portfolio
import numpy  
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

__all__ = []
__version__ = 0.1
__date__ = '2016-06-29'
__updated__ = '2016-06-29'

DEBUG = 1
TESTRUN = 0
PROFILE = 0


def MA(df, n):
    MA = pd.Series(pd.rolling_mean(df['Close'], n), name='MA_' + str(n))
    df = df.join(MA)  
    return df


# MACD, MACD Signal and MACD difference
def MACD(df, n_fast, n_slow):

    EMAfast = pd.Series.ewm(df['Close'], ignore_na=False, span=n_fast, min_periods=n_slow - 1, adjust=True).mean()
    EMAslow = pd.Series.ewm(df['Close'], ignore_na=False, span=n_slow, min_periods=n_slow - 1, adjust=True).mean()
    MACD = 'MACD_' + str(n_fast) + '_' + str(n_slow)
    df[MACD] = pd.Series(EMAfast - EMAslow, name = MACD)

    MACDsignal = pd.Series(pd.Series.ewm(df[MACD], ignore_na=False, span=9, min_periods=8, adjust=True).mean(),
                           name='MACDsignal_' + str(n_fast) + '_' + str(n_slow))

    MACDindicator = pd.Series(df[MACD] - MACDsignal, name='MACDindicator_' + str(n_fast) + '_' + str(n_slow))

    df = df.join(MACDsignal)
    df = df.join(MACDindicator)
    return df

def func(df, portfolio, explain=False):
    x = portfolio.transaction(df['Close'], df['MACDindicator_12_26'], explain)
    return pd.Series(dict(Cash=portfolio.getCash(), Shares=portfolio.getShares()))

def func2(df, portfolio):
    portfolio.transaction(df['Close'], df['MACDindicator_12_26'])
    return pd.Series(dict(Cash=portfolio.getCash(), Shares=portfolio.getShares()))
    
def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    # Setup argument parser
    parser = ArgumentParser(description='', formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-r", "--recursive", dest="recurse", action="store_true", help="recurse into subfolders [default: %(default)s]")
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")

    data = web.DataReader('ORCL', data_source='yahoo')
    macd = MACD(data, 12, 26)
    
    bestValue = 0
    bestSellBias = 0
    bestBuyBias = 0
                
#     portfolio = Portfolio.Portfolio(10000, 0.20, -0.50)
#     x = macd.apply(func=func, axis=1, reduce=False, args=([portfolio]))
#     macd = pd.concat([macd, x], axis=1)
#     
#     macd[['Close', 'MACDindicator_12_26', 'Shares', 'Cash']].plot(subplots=True, color='blue')
#     plt.grid()
#     plt.show()    
     
    for buyBias in range(0, 10, 1):
        for sellBias in range(0, -10, -1):
            portfolio = Portfolio.Portfolio(10000, float(buyBias) / 10, float(sellBias) / 10)

            value = portfolio.transactions(macd['Close'], macd['MACDindicator_12_26'])
 
            if value > bestValue:
                bestValue = value
                bestBuyBias = float(buyBias) / 10
                bestSellBias = float(sellBias) / 10
                    
            #print "%5.2f %5.2f %.2f" % (float(buyBias)/10, float(sellBias)/10, value)
                
        
    print "best %.2f %.2f %.2f" % (bestValue, bestBuyBias, bestSellBias)
        
    portfolio = Portfolio.Portfolio(10000, bestBuyBias, bestSellBias)
    # Get the MACD and apply the portfolio function to it, getting a another series.
    x = macd.apply(func=func, axis=1, reduce=False, args=([portfolio, True]))
    # Concatenate these series together 
    macd = pd.concat([macd, x], axis=1)
    
    macd[['Close', 'MACDindicator_12_26', 'Shares', 'Cash']].plot(subplots=True, color='blue')
    plt.grid()
    plt.show()          

    return 0

if __name__ == "__main__":        
    sys.exit(main())