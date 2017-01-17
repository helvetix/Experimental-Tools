import numpy as np
import sys
import math
import portfolio
import pandas as pd
import pandas_datareader.data
import matplotlib.pyplot as plt
from numpy.fft.helper import fftfreq
from scipy.optimize.optimize import Result

#
# Doesn't the FFT also need to know the number of points in one period?
# I guess that's the length of the original FFT array minus 1 times two...


def _compute_amplitude(x, y):
    return math.sqrt((x * x) + (y * y))


def normalize(fft, precision=0):
    nPointsPerPeriod = len(fft)
    print "normalize: fft", fft

    scale = len(fft) / 2
    bias = np.abs(fft[0]) / nPointsPerPeriod
    result = [nPointsPerPeriod, (0, round(bias, precision), 0)]

    for i in range(1, len(fft)):
        amplitude = round(np.abs(fft[i]), precision)
        print "normalize: ", amplitude, np.abs(fft[i])

        if amplitude > 0.0:
            result.append((fft[i][0], round(fft[i][1], precision), round(fft[i][2], precision)))
    print "normalize: result", result
    return result


class FFT:
    def __init__(self, rfft, zeroThreshold=0.0, precision=20):
        self.rfft = rfft

        self.nPointsPerPeriod = (len(rfft) - 1) * 2
        self.bias = round(np.abs(self.rfft[0]) / (len(self.rfft) - 1) / 2, precision)
        return

    def get_fft(self, precision=0):
        return [np.around(x, precision) for x in self.rfft]

    def _compute_amplitude(self, x, y):
        return math.sqrt((x * x) + (y * y))

    def normalize(self, precision=0):
        print "normalize: fft", self.fft
        result = [self.nPointsPerPeriod, (0, round(self.bias, precision), 0)]

        for i in range(2, len(self.fft)):
            amplitude = self._compute_amplitude(round(self.fft[i][1], precision), round(self.fft[i][2], precision))
            print "normalize: ", amplitude, self.fft[i][1], round(self.fft[i][1], precision), self.fft[i][2], round(self.fft[i][2], precision)

            if amplitude > 0.0:
                result.append((self.fft[i][0], round(self.fft[i][1], precision), round(self.fft[i][2], precision)))
        print "normalize: result", result
        return result

    def rfftNormalize(self, zeroThreshold=0.0):

        result = [self.nPointsPerPeriod, (0, self.bias, 0)]
        scale = len(self.rfft) - 1

        for i in range(1, len(self.rfft)):
            realAmplitude = self.rfft[i].real/scale
            imagAmplitude = self.rfft[i].imag/scale * -1

            amplitude = math.sqrt((self.rfft[i].real * self.rfft[i].real) + (self.rfft[i].imag * self.rfft[i].imag)) /scale
            if amplitude <= zeroThreshold:
                amplitude = 0.0
            if amplitude > 0.0:
                result.append((i, imagAmplitude, realAmplitude))
        return result

    def derivative(self, precision=20):
        result = [np.complex(0, 0)]
        for e in self.rfft[1:]:
            e = np.around(np.complex(-e.imag, e.real), precision)
            result.append(e)
        return result

    def rfftDerivative(self):
        result = []

        # F ( A cos(Fx) - B sin(Fx))
        #
        # 1 ( 1 cos(Fx) - 0 sin(Fx)) => cos(Fx)
        # 2 ( 1 cos(2x) - 0 sin(2x)) => 2cos(2x)
        # tuple = (F, -B*F, A*F)

        for i in range(0, len(self.fft)):
            p = self.rfft[i]
            result.append(np.complex(-self.rfft[i].imag, self.rfft[i].real))

        return result

    def __repr__(self):
        return str(self.rfft)


def loadLines():
    lines = sys.stdin.readlines()
    array = np.array(map(lambda line: float(line), lines))
    return array


# I want to specify the values ranging from 0 to the length of one period.
# The number of values is the total number of samples across the period.
# So a length of 100 is 1 period with 100 samples.
#
# What I need is the ability to generate a single sample that represents a day.
# Have it oscillate over several days.
# 
# For a stock price, it gets sampled once per day, it oscillates over several days.
#
def pointGenerator(FFT, index):
    bias = math.sqrt(FFT[1][1] * FFT[1][1] + FFT[1][2] * FFT[1][2])
    result = bias

    for fft in FFT[2:]:
        point = fft[0] * 2*np.pi * index/FFT[0]

        sine = math.sin(point) * fft[1]
        cosine = math.cos(point) * fft[2]

        result = result + sine + cosine

    return result


def singleWaveGenerator(numberOfSamples, frequency):
    result = pd.Series(np.linspace(start=0, stop=frequency*(2*np.pi), num=numberOfSamples, endpoint=True))
    return result


# Generate a compound wave given the coefficients in the FFT.
def waveGenerator2(FFT):
    numberOfSamples = FFT[0]

    bias = math.sqrt(FFT[1][1] * FFT[1][1] + FFT[1][2] * FFT[1][2])
    result = np.empty(numberOfSamples)
    result.fill(bias)
    for fft in FFT[2:]:
        wave = singleWaveGenerator(numberOfSamples, fft[0])
        sine = np.sin(wave) * fft[1]
        cosine = np.cos(wave) * fft[2]

        result = pd.Series(result + sine + cosine)

    return result


def waveGeneratorDerivative(FFT):
    numberOfSamples = FFT[0]
    bias = 0.0
    result = np.empty(numberOfSamples)
    result.fill(bias)

    for fftPoint in FFT[1:]:
        wave = singleWaveGenerator(numberOfSamples, fftPoint[0])
        # result = F ( A cos(Fx) - B sin(Fx))
        e1 = fftPoint[0] * np.cos(wave) * fftPoint[1]
        e2 = fftPoint[0] * np.sin(wave) * fftPoint[2]
        result = pd.Series(result + (e1 - e2))

    return result


def rfftDerivative(rfft):
    result = [(0, 0.0, 0)]

# F ( A cos(Fx) - B sin(Fx))
#
# 1 ( 1 cos(Fx) - 0 sin(Fx)) => cos(Fx)
# 2 ( 1 cos(2x) - 0 sin(2x)) => 2cos(2x)
# tuple = (F, -B*F, A*F)

    for i in range(1, len(rfft)):
        p = rfft[i]
        result.append(tuple([p[0], -p[2]*p[0], p[1]*p[0]]))

    return result


def fft_derivative(simple_fft):
    result = [numberOfPoints, (0, 0.0, 0.0)]
    for e in simple_fft[1:]:
        t = (e[0], e[2], e[1])
        result.append(t)

    return result


numberOfPoints = 32
bias = (0, 100.0, 0.0)
actualFFT = [numberOfPoints, bias, (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (4.0, 1.0, 1.0)]
actualFFT = [numberOfPoints, bias, (1.0, 1.0, 0.0)]
actualFFT = [numberOfPoints, bias, (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)]
actualDerivativeFFT = [numberOfPoints, (0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]

actualDerivativeFFT = fft_derivative(actualFFT)

if True:
    print actualDerivativeFFT
    sample = waveGenerator2(actualFFT)
    sampleDerivative = waveGenerator2(actualDerivativeFFT)
    sampleDerivativeFFT = np.fft.rfft(sampleDerivative)
    print "actualFFT spec", actualFFT
    print "derivativeFFT", len(sampleDerivativeFFT), sampleDerivativeFFT
else:
    data = pd.DataFrame.from_csv('orcl-2015.csv')
    sample = data['Close'][0:numberOfPoints]

fft = FFT(np.fft.rfft(sample))

print "raw FFT", fft
print "rounded FFT", fft.get_fft(0)

print

print "derivative", [np.around(a) for a in sampleDerivativeFFT]
print "computed  ", fft.derivative(0)


#generated = waveGenerator2(normalizedFFT)
generated = np.fft.irfft(fft.rfft)
reducedGenerated = np.fft.irfft(fft.rfft)

plt.grid()
plt.plot(sample)
plt.plot(sampleDerivative)

plt.figure()
plt.grid()
# plt.plot(generated)
plt.plot(reducedGenerated)

# derivative = waveGeneratorDerivative(normalizedFFT)
# 
# computedDerivativeFFT = fft.derivative()
# generatedDerivative = waveGenerator2(computedDerivativeFFT)

plt.figure()
#plt.hist(np.abs(fft.rfft)[1:], rwidth=.1)

x_indicies = np.arange(len(np.abs(fft.rfft)[1:]))
x_indicies = np.arange(1, len(np.abs(fft.rfft)[1:]) + 1)
plt.bar(x_indicies, np.abs(fft.rfft)[1:], 0.35, color='r')
# plt.plot(generatedDerivative)
# plt.grid()

plt.show()

if False:
    x = []
    
    portfolio = portfolio.Portfolio(10000.0)
    
    offset = 0
    if True:
        for i in range(0, numberOfPoints):
            point = offset + i
            predictedPrice = pointGenerator(normalizedFFT, point)
    
            indicator = pointGenerator(computedDerivativeFFT, point)
            x.append(indicator)
            portfolio.transaction(predictedPrice, indicator, explain=True)
    else:
        for i in range(0, len(data['Close'][numberOfPoints:])):
            point = offset + i
            predictedPrice = pointGenerator(normalizedFFT, point)
            actualPrice = data['Close'][numberOfPoints:][point]
    
            indicator = pointGenerator(computedDerivativeFFT, point)
            x.append(indicator)
            portfolio.transaction(actualPrice, indicator, explain=True)
    
    print portfolio, portfolio.getValue()

# plt.figure()
# plt.plot(x)
# plt.grid()
# plt.show()
