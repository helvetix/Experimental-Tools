'''
Created on Jul 1, 2016

@author: gscott
'''

import sys
import os

import portfolio as Portfolio
import numpy  as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

def function(x, y, z, h):
    print type(x), x, type(y), y, type(z), z, type(h), h
    return -1


def main():

    data = pd.DataFrame({
                       'A': np.linspace(0, 10, endpoint=False, num=10),
                       'B': np.linspace(10, 20, endpoint=False, num=10),
                       'C': np.linspace(20, 30, endpoint=False, num=10), 
                       })
    
    data = data.head(10)
    #pd.rolling_apply(data, 2, function, 0, None, False, ('a', 'b'), {"h": 2})
    print pd.DataFrame.rolling(data, center=False,min_periods=0,window=2).apply(args=('a', 'b'), func=function, kwargs={"h": 2})
    
    #pd.DataFrame.rolling(data, 2, function, 0, None, False).apply(('a', 'b'), {"h": 2})
    print data
    return


if __name__ == '__main__':
    main()
    pass