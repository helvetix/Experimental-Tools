import numpy as np
from scipy.fftpack import fft
import pandas as pd
import pandas.io.data as web

# Number of sample points
N = 720
x = np.linspace(0.0, 2.0*np.pi, N, False)

y = np.sin(50.0 * x)

for i in range(0, len(y)):
    print(y[i])
    
yf = fft(y)

r = np.abs(yf)
import matplotlib.pyplot as plt

plt.plot(r)
plt.grid()
plt.show()
