'''
Created on Jun 22, 2016

@author: gscott
'''
from scipy.odr.odrpack import Output

import sys
import string
import pprint
import ast

class tick(dict):
    
    def __init__(self, timestamp, openPrice, highPrice, lowPrice, closePrice, volume, splitFactor=None, earnings=None, dividends=None):
        super(tick, self).__init__()
        self["Timestamp"] = timestamp
        self["Open"] = float(openPrice)
        self["High"] =  float(highPrice)
        self["Low"] = float(lowPrice)
        self["Close"] = float(closePrice)
        self["Volume"] = int(volume)
        return
    
    def __str__(self):
        return self.__repr__()
            
    def __cmp__(self, other):
        return self["Timestamp"] - other["Timestamp"]
    
    def timestamp(self):
        return self["Timestamp"]


class time_series(list):
    def __init__(self):
        super(time_series, self).__init__()
        return
    
def putFile(output):
    x = sorted(self)
    pprint.pprint(x)
    return
        
def getFile(input):
    return ast.literal_eval(input.read())
                    
