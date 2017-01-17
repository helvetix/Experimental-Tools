import sys
import string
import quantquote
import pprint
import argparse
   
def select_keys(dictionary, keys):
     result = { key: dictionary[key] for key in keys }
     return result
 
 
def select_in_range(tick, start, stop):
    return tick["Timestamp"] >= start and tick["Timestamp"] <= stop
    
         
if __name__ == '__main__':
    '''
    '''
    
    parser = argparse.ArgumentParser(prog='select', formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    parser.add_argument('--start', default=0, action="store", help="Select the starting date of ticks.")
    parser.add_argument('--stop', default=99999999999999, action="store", help="Select the stopping date of ticks.")
    parser.add_argument("keys", help="Keys to extract.", nargs="*")
    parser.add_argument('--text', default=False, action='store_true', required=False, help="Display space separated fields.")
        
    args = parser.parse_args()    
    start = int(args.start)
    stop = int(args.stop)
    
    time_series = quantquote.getFile(sys.stdin)
    series = filter(lambda e: select_in_range(e, start, stop), time_series)
    series = map(lambda e: select_keys(e, args.keys), series)
        
    
    if args.text:
        for e in series:
            for key in args.keys:
                print e[key],
            print
    else:
        pprint.pprint(series)
    