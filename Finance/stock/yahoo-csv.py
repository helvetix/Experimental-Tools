import urllib2, urllib, json, sys
import csv
import StringIO
import time
from fileinput import filename
import quantquote

def fileReader(filename):
    with open(filename, 'r') as f:
        result = f.read()
        
    return result

def yahooFetcher(ticker):
    baseurl="http://real-chart.finance.yahoo.com/table.csv?s=%s&d=5&e=21&f=2016&g=d&a=2&b=12&c=1986&ignore=.csv" % (ticker)
    result = urllib2.urlopen(baseurl).read()
    return result

def timeToInt(t):
    return t.tm_year * 10000000000 + t.tm_mon * 100000000 + t.tm_mday * 1000000 + t.tm_hour * 1000 + t.tm_min * 10 + t.tm_sec


if __name__ == '__main__':
    '''
    '''
    
    #result = yahooFetcher("ORCL")
    result = fileReader("/Users/gscott/Downloads/oracl.csv")
    xx = csv.reader(StringIO.StringIO(result))

    time_series = quantquote.time_series()
    for row in xx:
        if row[0] != "Date":
            d = time.strptime(row[0], "%Y-%m-%d")
            time_series.append(quantquote.tick(timeToInt(d), row[1], row[2], row[3], row[4], row[5]))
 
    x = sorted(time_series)
            
    quantquote.putFile(time_series, sys.stdout)
