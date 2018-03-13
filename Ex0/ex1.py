# coding: utf-8

# import Python modules task2
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack as fft
#task1
print('hello world')
# Read the wavefile: sound of plucking a guitar string
fs,x = wavfile.read('Wang Shanshan.mp3.m4a')

# For Windows, you can use winsound to play the audio file
# uncomment the following two line if you are on windows and want to play the audio file
# import winsound
# winsound.PlaySound('gtr55.wav', winsound.SND_FILENAME)

# Your code here: print out type, length of x, length of audio signal in seconds
# Hint:help()
print(type(x))
print(len(x))
print(len(x)/fs)

# task 3
# Time-domain visualization of the signal
# Your code here: make the x-axis or time scale, should be same shape as x.
# Hint: use numpy.linspace to make an array containing the numbers you want

t = np.linspace(0, len(x)/fs, num=len(x))
# plotting
plt.subplot(3, 1, 1)
plt.plot( t, x )
plt.title('Time domain visualization of the audio signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.grid('on')

# Frequency domain visualization of the signal, logarithmic magnitude
# Your code here: frequency scale, fft with scipy
# Hint: Nyquist frequency, help()
max_freq = fs/2
X = fft.fft(x)
winlen = 1024
frq_scale = np.linspace(0, max_freq, winlen/2.0-1)
mag_scale = 20.0*np.log10(np.abs(X[0:winlen//2-1]))
# plotting
plt.subplot(3, 1, 2)
plt.plot( frq_scale, mag_scale)
plt.title('Frequency domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
#
#
# # Your code here: Spectrogram of the audio signal
f,t,X = scipy.signal.spectrogram(x, fs)
mag_scale1 = 20*np.log10(1e-6+np.abs(X))
# plotting
plt.subplot(3, 1, 3)
plt.pcolormesh(t,f, mag_scale1)
plt.xlabel('time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Log-magnitude spectrogram')
plt.colorbar()

# Show the figure.
plt.tight_layout()
plt.show()

# Bonus points: record your own audio file; save the figure

