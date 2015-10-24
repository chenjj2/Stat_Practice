'''
signal: sin wave with amplitude 1
noise: white noise with amplitude orders of magnitude larger than the signal
question: can we find the signal with fft?
'''

''' adjust from example given on http://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python '''
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Signal/Noise
s2n = 5e-1

# Number of samplepoints
N = 6e2
# sample spacing
T = 1.0 / 200.0
x = np.linspace(0.0, N*T, N)
y = s2n * np.sin(50.0 * 2.0*np.pi*x) + np.random.normal(0.,1.,N)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

col=2
row=1

fig, (ax1, ax2) = plt.subplots(row,col,figsize=(col*5,row*5))
ax1.plot(x,y)
ax2.plot(xf, 2.0/N * np.abs(yf[0:N/2]))

plt.savefig('plt_fft_logsignal.png')