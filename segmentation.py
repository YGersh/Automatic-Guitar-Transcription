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
import utilities

hop_length=100
def novelty(x, sr):  # generates a novelty function, plots if 3rd arg=1
    onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=100)
    return onset_env

def onset_samples_detect(x, sr):  # detect samples of onset BACKTRACKING ENABLED
    onset_samples = librosa.onset.onset_detect(x, sr=sr, units='samples', hop_length=hop_length, backtrack=True,
                                               pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=0)
    return onset_samples

def onset_pad(onset_samples, x, sr):  # include stard and end of signal in onsets
    onset_bounds = np.concatenate(
        [[0], onset_samples, [len(x)-1]])  # include start and end of signal as onsets of notes
    return onset_bounds

def onset_to_time(onset_bounds, sr):
    onset_times = librosa.samples_to_time(onset_bounds, sr=sr)
    return onset_times

def onset_detect(x,sr):  # single func to call to get time onsets
    onsets=onset_to_time(onset_pad( onset_samples_detect(x,sr),x, sr), sr)
    return onsets




