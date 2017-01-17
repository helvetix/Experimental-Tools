import numpy as np
import sys
import json
from pprint import pprint

lines = sys.stdin.readlines()

array = np.array(map(lambda line: float(line), lines))
 
import matplotlib.pyplot as plt
plt.plot(array)
#splt.yscale('log')
plt.grid()
plt.show()
