import numpy as np
from scipy.fftpack import fft
import sys

# Number of sample points
N = int(sys.argv[1])
x = np.linspace(0.0, 2.0*np.pi, N, False)

y = np.sin(50.0 * x)

for i in range(0, len(y)):
    print(y[i])
    
