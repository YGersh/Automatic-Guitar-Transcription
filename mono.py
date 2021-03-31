from __future__ import division
import matplotlib
import seaborn
import numpy as np
import numpy.fft as fft
import scipy, scipy.fftpack, IPython.display as ipd, matplotlib.pyplot as plt
import librosa, librosa.display
#import simpleaudio as sa
import xml.etree.cElementTree as ET
import peakutils as pk
import copy
from numpy import polyfit, arange
from numpy.fft import rfft, fft, ifft, irfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal.windows import blackmanharris
from klapuri import F0Estimate as F0Estimate
import segmentation
import utilities

def pitch_detect_autocorellation(segment, sr, fmin=40, fmax=1400): #pitch detect using autocorellation
    """
        Pitch detection using Autocorellation
        Inputs: audio segment, sampling rate of segment
        Peak detection of autocorellation done using peakutils, as standard argmax was inaccurate
    """
    r = librosa.autocorrelate(segment)
    i_min = sr / fmax
    i_max = sr / fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
   # i = r.argmax()
   #  f0 = float(sr) / i
    i_peak = pk.indexes(r, thres=0.8, min_dist=5)[0] #use peakutils for more robust peak detection, since numpy is not precise enough
    i_interp = utilities.parabolic(r, i_peak)[0]
    f0=float(sr)/i_interp
    return f0

def pitch_detect_HPS(x, sr, depth):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    """
    windowed = x * blackmanharris(len(x))
    from pylab import subplot, plot, log, copy, show
    # harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = depth
    freq=[]
    for x in range(2, maxharms):
        a = copy(c[::x])  # Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = argmax(abs(c))
        true_i = utilities.parabolic(abs(c), i)[0]
        freq.append(sr*true_i/len(windowed))
        c *= a
    f0=np.mean(freq)
    return int(f0)
