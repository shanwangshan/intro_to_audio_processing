# coding: utf-8

# import Python modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd
import math


## 1.  (0.5 pt in total)
# Edit the script to analyse the audio wav file (short recording of a musical instrument) in time domain and frequency domain using python modules
# you can resuse parts of exercise0.py from your previous exercise
#!! FILL IN PARTS WITH "None"
plt.figure(1)
# Plot time domain visualization 
fs,x = wavfile.read('glockenspiel-a-2.wav')  #fs is the sampling frequency of the signal
fs = float(fs)
# the length of audio signal in seconds
print(float(len(x)/fs))
t_scale = np.linspace(0, len(x)/fs, num=len(x))
amp_scale = x

plt.subplot(2, 1, 1)
plt.plot( t_scale, amp_scale)
plt.title('Time domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')

# Plot frequency domain visualization using FFT
#!! FILL IN PARTS WITH "None"  
max_freq = fs/2
winlen = 1024*4
X = scipy.fft(x[10000:10000+winlen])
frq_scale = np.linspace(0, max_freq, winlen/2.0-1)
magnitude = 20*np.log10(np.abs(X[0:int(winlen/2.0)-1]))

plt.subplot(2, 1, 2)
plt.plot( frq_scale, magnitude)
plt.title('Frequency domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig('figure1.jpg')
plt.show()
print(frq_scale[np.argmax(magnitude)])
## 2  (0.5 pt in total)
# Edit the script to generate a Sine wave signal y1, 
# with amplitude 3000 (so you will hear something if you play the signal),
# frequency F0, and length of t seconds, as calculated from step 1.

# Plot Sine wave, y1 (0.2 pt).
#!! FILL IN PARTS WITH "None"
plt.figure(2)
# # # Plot time domain visualization
t_scale =np.linspace(0, len(x)/fs, num=len(x))
y1 = amp_scale = 3000*scipy.sin(2*np.pi*1875*t_scale)

plt.subplot(2, 1, 1)
plt.plot(t_scale, y1)
plt.title('Time domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
#
# # Apply fast Fourier transform (FFT) to the Sine wave. Display the FFT of the waveform (0.2 pt).
# #!! FILL IN PARTS WITH "None"
# # Plot frequency domain visualization using FFT
X1 = scipy.fft(y1[10000:10000+winlen])
frq_scale1 = np.linspace(0, max_freq, winlen/2.0-1)
magnitude1 = 20*np.log10(np.abs(X1[0:int(winlen/2.0)-1]))
#
plt.subplot(2, 1, 2)
plt.plot( frq_scale1, magnitude1)
plt.title('Frequency domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig('figure2.jpg')
plt.show()
#
# # Play its sound on the computer (0.1 pt).
# #!! FILL CODE HERE
print('play y1')
sd.play(y1,fs)
#
# # For example on windows, you would save the Sine wav file and play it as following
# # scipy.io.wavfile.write('y1.wav', fs, np.int16(y1))
# # winsound.PlaySound('y1.wav', winsound.SND_FILENAME)
#
# ## 3 (0.5 pt in total)
# # Edit the script to generate another Sine wave, y2
# # y2 has the same property as y1, except that y2 has a new frequency of F1.
# # F1 equals to the second partial of the audio wav file <glockenspiel-a-2.wav>.
# # make a new signal by summing y1 and y2 together: y = y1 + 0.5*y2.
# # Plot y time domain and frequency domain visualizations. Display the plots on screen (0.2 pt).
#
plt.figure(3)
t_scale =np.linspace(0, len(x)/fs, num=len(x))
y2=amp_scale = 3000*scipy.sin(2*np.pi*7000*t_scale)
sd.sleep(4000)
print('play y2')
sd.play(y2,fs)
y = y1 + 0.5*y2
sd.sleep(4000)
print('play y')
sd.play(y,fs)
# # Plot time domain visualization
plt.subplot(2, 1, 1)
plt.plot(t_scale, y)
plt.title('Time domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
# # Plot frequency domain visualization using FFT
X2 = scipy.fft(y[10000:10000+winlen])
frq_scale2 = np.linspace(0, max_freq, winlen/2.0-1)
magnitude2 = 20*np.log10(np.abs(X2[0:int(winlen/2.0)-1]))
plt.subplot(2, 1, 2)
plt.plot(frq_scale2, magnitude2)
plt.title('Frequency domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('figure3.jpg')
plt.show()
#
# ## 4 (0.5 pt in total)
# # Compare the plots in step 2 and step 3
# # Question: play y1, y2, y. listen to the signals, do you think your synthesized signal resembles the original musical instrument (0.2 pt)?
# # Question: what would you do to make the synthesized signal more similar to the original (0.3 pt)?
#

