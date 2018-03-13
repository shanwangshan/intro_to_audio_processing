#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 10:32:30 2018

@author: wangshanshan
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd


fs, x = wavfile.read('rhythm_birdland.wav')  
fs = float(fs)
f, t, Fhi = signal.stft(x, fs, nperseg=1000)
plt.figure(1)
plt.pcolormesh(t, f, np.abs(Fhi))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.savefig('the original spectrogram')
gamma = 0.3
alpha = 0.3
Whi = (np.abs(Fhi)) ** (2 * gamma) #a range-compressed version of the power spectrogram
Phi = (1 / 2) * Whi #initialize the values of P
Hhi = (1 / 2) * Whi #initialize the values of H
kmax = 10
for k in range(0, (kmax - 1)):
    for h in range(1, (np.shape(Hhi)[0]) - 1):
        for i in range(1, (np.shape(Hhi)[1]) - 1):
            delta = alpha * ((Hhi[h, i - 1] - 2.0 * Hhi[h, i] +  Hhi[h, i + 1]) / 4.0) \
            - (1 - alpha) * ((Phi[h - 1, i] - 2 * Phi[h, i] + Phi[h + 1, i]) / 4.0)
            Hhi[h, i] = np.minimum(np.maximum((float(Hhi[h, i]) + float(delta)), 0.0), float(Whi[h, i]))
            Phi[h, i] = Whi[h, i] - Hhi[h, i]
# At the end we will have H kmax-1 and P kmax-1
print("10 iterations have finished")
#binarize the separation
for h in range(0, np.shape(Hhi)[0]):
    for i in range(0, np.shape(Hhi)[1]):
        if Hhi[h, i] < Phi[h, i]:
            Hhi[h, i] = 0
            Phi[h, i] = Whi[h, i]
        elif Hhi[h, i] >= Phi[h, i]:
            Hhi[h, i] = Whi[h, i]
            Phi[h, i] = 0
print("Matrix has been constructed")
magH = np.abs(Hhi**(1/(2*gamma)))
Hreal = magH*(np.cos(np.angle(Fhi)))
Himag = 1j*magH*(np.sin(np.angle(Fhi)))
HIFT = Hreal + Himag
t1, h = signal.istft(HIFT, fs, nperseg=1000)
plt.figure(2)
plt.plot(t1, h)
plt.title('Time domain visualization of Harmonic signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.show()
plt.savefig('the time domain harmonic signal')
plt.figure(3)
magP = np.abs(Phi**(1/(2*gamma)))
Preal = magP*(np.cos(np.angle(Fhi)))
Pimag = 1j*magP*(np.sin(np.angle(Fhi)))
PIFT = Preal + Pimag
t, p = signal.istft(PIFT, fs, nperseg=1000)
plt.plot(t, p)
plt.title('Time domain visualization of Percussive signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.show()
plt.savefig('the time domain percussive signal')

y=(h+p)
x=np.array(x)
error=x-y[0:441001]
SNR=10*np.log10(np.sum(x**2)/np.sum(error**2))
print('the SNG value is ' ,SNR)
plt.figure(4)
f, t, HS = signal.stft(h, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(HS), vmin=0)
plt.title('Harmonic spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.savefig('the harmonic spectrogram')
plt.figure(5)
f, t, PS = signal.stft(p, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(PS), vmin=0)
plt.title('Percussive spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.savefig('the percussive spectrogram')
plt.figure(6)
plt.plot(t1[0:441001], error)
plt.title('error signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.show()
plt.savefig('erorr signal')
#play the sound
print('playing the original one')
sd.play(x,fs)
sd.sleep(9000)
scipy.io.wavfile.write('Percussive.wav', int(fs), np.int16(p))
fs, p = wavfile.read('Percussive.wav')
print('playing the percussive one')
sd.play(p, fs)
sd.sleep(9000)
scipy.io.wavfile.write('Harmonic.wav', int(fs), np.int16(h))
fs, h = wavfile.read('Harmonic.wav')
print('playing the harmonic one')
sd.play(h, fs)
