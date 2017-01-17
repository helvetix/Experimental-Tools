'''
Created on Jun 29, 2016

@author: gscott
'''
import math
import sys
import string
import pprint
import argparse
import pandas as pandas

class Portfolio(object):
    '''
    classdocs
    '''

    def __init__(self, cash, buyBias=0.5, sellBias=0.0):
        '''
        Constructor
        '''
        self.history = pandas.DataFrame([ ( cash, 0) ], columns=['Cash', 'Shares'])
        self.shares = 0
        self.cash = cash
        self.buyBias = buyBias
        self.sellBias = sellBias
        self.lastPrice = 0.0
        return
    
    def getValue(self):
        return self.shares * self.lastPrice + self.cash
    
    def getShares(self):
        return self.shares
    
    def getCash(self):
        return self.cash
    
    def transactions(self, price, signal):
        for index in range(0, price.size):                    
            self.transaction(price[index], signal[index])
            
        return self.getValue()

                
    def transaction(self, price, signal, explain=False):
        if signal != math.isnan(signal):        
            if signal > self.buyBias: # Buy
                sharesToBuy = int(self.cash / price)
                if sharesToBuy > 0:
                    self.shares = int(self.shares + sharesToBuy)
                    self.cash = self.cash - (price * self.shares)
                    if explain:
                        print "Buy %d $%.2f" % (sharesToBuy, self.cash)
            elif signal < self.sellBias: # Sell
                sharesToSell = self.shares
                if sharesToSell > 0:
                    self.cash = self.cash + (sharesToSell * price)
                    self.shares = 0
                    if explain:
                        print "Sell %d $%.2f" % (sharesToSell, self.cash)
            self.lastPrice = price
            return self.getValue()
        return 0
    
    def plot(self):
        self.history[['Cash', 'Shares']].plot(subplots=True, color='blue', figsize=(8, 6)) 
        
    def __str__(self):
        return "Portfolio: shares=%d cash=%.2f" % (self.shares, self.cash)
           
if __name__ == '__main__':
    '''
    '''
    
    parser = argparse.ArgumentParser(prog='select', formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    parser.add_argument('--cash', default=1000, action="store", help="Select the starting date of ticks.")
        
    args = parser.parse_args()
    
    portfolio = Portfolio(args.cash)
    
    portfolio.transaction(10.0, 1)
    print portfolio
    print portfolio.getValue()
