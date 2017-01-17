import numpy as np
import sys
import pandas as pd
import pandas_datareader.data
import matplotlib.pyplot as plt

def loadLines():
    lines = sys.stdin.readlines()
    array = np.array(map(lambda line: float(line), lines))
    return array

# I want to put in the frequency in days, with each day being a sample.
# So a frequency of 4 days means it goes from 0 to 360 in 4 days with 4 samples in the array.
# A frequency of 1 day means 0-360 in 1 period, which is a noop.
# If the samples are once per day, then there are 4 samples spaced 360/4 apart.
#
# Then there total days, or nPeriods which is the the frequency * nPeriods in samples.
#
# The total number of samples is the frequencyInDays * nPeriods. 
#
# The linspace starts at 0 and stops at 360 * nPeriods
    
def sineWaveGenerator(frequencyInDays, nPeriods, scale = 1):
    bias = 1.0
    stop = nPeriods * 360
    num = frequencyInDays * nPeriods
    
    values = np.linspace(start=0, stop=stop, num=num, endpoint=False)

    wave1 = np.sin(values * np.pi / 180.)  * 2.0
    wave2 = np.sin(2.0 * values * np.pi / 180.)
    
    result = pd.Series((wave1 + wave2) + bias)
    return result
    
def sineWaveGenerator2(fftList, index):
    bias = fftList[0][1]
        
    result = bias

    for t in fftList[1:]:
        point = (360.0 / t[0]) * index * np.pi / 180.0
        value = np.sin(point) * t[1]
        result = result + value            
        
    return result

def sineWaveGenerator3(fftArray, index):
    print "sineWaveGenerator3", index
    bias = fftArray.real[0]
        
    result = 0.0
    for i in range(1, len(fftArray)):
        # print i, fftArray.real[i], fftArray.imag[i]
        frequency = len(fftArray) / index
        #print "frequency", frequency
        point = i * index
        result += fftArray.real[i]*np.cos(point) + fftArray.imag[i]*np.sin(point)
            
    return result


def sineWaveGeneratorDerivative(fftList, index):
        
    result = 0.0
    for t in fftList[1:]:
        point = (360.0 / t[0]) * index
        value = np.cos(point * np.pi / 180.0) * t[1]
            
        result = result + value
        
    return result

def getFFTBias(fftArray):
    return fftArray[0]/(len(fftArray) - 1) / 2.0

def getFFTFrequency(fftArray, index):
    return (len(fftArray) - 1) * 2.0 / index

def getFFTSignal(fftArray, index): 
    fftArray[index]
    
def fftNormalizeAmplitude(amplitude):
    return

def fftNormalize(fftArray, zeroThreshold=0.0):
    result = [ (0, getFFTBias(fftArray)) ]
    for i in range(1, len(fftArray)):
        if fftArray[i] <= zeroThreshold:
            fftArray[i] = 0.0
        amplitude = fftArray[i]/(len(fftArray) - 1)
        result.append((getFFTFrequency(fftArray, i), amplitude))
    return result

numberOfSamples = 100
# actualFFT = [  (0, 1.0), (5.0, 2.0), (2.5, 1.0)  ]
actualFFT = [(0, 1.00), (100.0, 0.00), (50.0, 0.00), (33.333333333333336, 0.00), (25.0, 0.00), (20.0, 0.0), (16.666666666666668, 0.0), (14.285714285714286, 0.0), (12.5, 0.018161639568703133), (11.11111111111111, 0.021487667627600127), (10.0, 0.025336522731835034), (9.090909090909092, 0.02988980345788296), (8.333333333333334, 0.035415253340738397), (7.6923076923076925, 0.042326775028958069), (7.142857142857143, 0.051302422163911086), (6.666666666666667, 0.063538418828692669), (6.25, 0.081365913637064774), (5.882352941176471, 0.11002392994269465), (5.555555555555555, 0.16426726481971859), (5.2631578947368425, 0.30801305179498717), (5.0, 1.8833778478849319), (4.761904761904762, 0.47806133892235664), (4.545454545454546, 0.21487485461572395), (4.3478260869565215, 0.13908464254743044), (4.166666666666667, 0.10267114008067711), (4.0, 0.080945207486212742), (3.8461538461538463, 0.066223302903085979), (3.7037037037037037, 0.055316530006758645), (3.5714285714285716, 0.046640561690547588), (3.4482758620689653, 0.03929187016597168), (3.3333333333333335, 0.032683215004217302), (3.225806451612903, 0.026371631920508537), (3.125, 0.019957084564997462), (3.0303030303030303, 0.012998469723185897), (2.9411764705882355, 0.0049075612382645907), (2.857142857142857, 0.0052395118969857544), (2.7777777777777777, 0.019123731826779095), (2.7027027027027026, 0.040330285633361253), (2.6315789473684212, 0.0783479492590341), (2.5641025641025643, 0.16969239215152981), (2.5, 0.71584366168025271), (2.4390243902439024, 0.55817684442105742), (2.380952380952381, 0.23989007758902633), (2.3255813953488373, 0.1667876535324094), (2.272727272727273, 0.13471321048466495), (2.2222222222222223, 0.11705149039342645), (2.1739130434782608, 0.10621612471543546), (2.127659574468085, 0.099241296029360659), (2.0833333333333335, 0.094751053372912786), (2.0408163265306123, 0.092046902986937379), (2.0, 0.090771433700655921)]

values = pd.Series(np.linspace(start=0, stop=numberOfSamples, num=numberOfSamples+1, endpoint=True))
actual1 = pd.Series(values.apply(lambda x: sineWaveGenerator2(actualFFT, x)))
actual = sineWaveGenerator(5, 20)

#data = pd.DataFrame.from_csv('orcl-2015.csv')
#actual = data['Close'][0:200]
    
print np.version.version
print sys.path
print "number of points=", len(actual)
print "actual\n", actual[:10]
# print "actual1\n", actual1[:10]
yf = np.fft.fft(actual)
#yf = np.abs(yf)
print "FFT\n", yf
#print "fftfreq\n", np.fft.fftfreq(numberOfSamples)

print "FFT length=%d  bias=%f  max=%f  index=%d" % (len(yf),  getFFTBias(yf), np.max(yf[1:]), np.argmax(yf[1:])+1)
print "original frequency %d" % (getFFTFrequency(yf, np.argmax(yf[1:])+1) )

normalized = fftNormalize(yf)
print normalized
sorted_by_amplitude = sorted(normalized, reverse=True, key=lambda tup: tup[1])
print sorted_by_amplitude


#values = pd.Series(np.linspace(start=0, stop=len(actual), num=len(actual)+1, endpoint=True))
# generated = pd.Series(values.apply(lambda x: sineWaveGenerator2(normalized, x)))
values = pd.Series(np.linspace(start=1, stop=len(actual), num=len(actual)+1, endpoint=True))
generated = pd.Series(values.apply(lambda x: sineWaveGenerator3(yf, x)))
print "generated\n", generated[:10]
indicator = pd.Series(values.apply(lambda x: sineWaveGeneratorDerivative(normalized, x)))
print "indicator\n", indicator[:10]

print "%6s %10s %10s %10s" % ("index", "actual", "predicted", "indicator")
for i in range(0, 10):
    print "%6d %10.4f %10.4f %10.4f" % (i, actual[i], sineWaveGenerator2(normalized, i), sineWaveGeneratorDerivative(normalized, i))
    
#actual.plot(subplots=True, color='blue')
#generated.plot(subplots=True, color='green')
#indicator.plot(subplots=True, color='red')
plt.plot(actual)
plt.plot(generated)
#plt.plot(indicator)
plt.grid()
plt.show()
