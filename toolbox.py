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
from mono import pitch_detect_autocorellation, pitch_detect_HPS
import inout
import pyfid

class Event: #declare in format pitch, onsetSec, offsetSec, fretNumber, stringNumber, excitationStyle, expressionStyle
    def __init__(self, params) :
        self.pitch = float(params[0])
        self.onsetSec = float(params[1])
        self.offsetSec = float(params[2])
        self.fretNumber = float(params[3])
        self.stringNumber = float(params[4])
        self.excitationStyle = str(params[5])
        self.expressionStyle = str(params[6])



hop_length=100
filename_mono = 'DATASET/dataset3/audio/pathetique_mono.wav'
xmlfile_mono = 'DATASET/dataset3/annotation/pathetique_mono.xml'
filename_poly = 'DATASET/dataset3/audio/pathetique_poly.wav'
xmlfile_poly = 'DATASET/dataset3/annotation/pathetique_poly.xml'

x, sr = librosa.load(filename_mono)
x_p, sr_p=librosa.load(filename_poly)


def extractor(x_m, x_p, sr_m, sr_p, select): #make event struct w extracted params
    onsets = segmentation.onset_detect(x_m, x_p, sr_m, sr_p)[select]
   # pitch = [0] #replace with pitch detect
    #excitation = 'X' #replace with excitation detect
    #offset = [0] #replace with offset detect
    #fret = [0] #replace with fret detection
    #string = [0] #replace with string detection
    #expression = 'X' #replace with expression detection
    X_events=[]
    for i in range(len(onsets)):
        current_event=[]
        current_event.append(0) #pitch
        current_event.append(onsets[i])
        current_event.append(0) #offset
        current_event.append(0) #fret
        current_event.append(0) #string
        current_event.append(0) #excitation
        current_event.append(0) #expression
        X_events.append(Event(current_event))
    return X_events


'''def errorcheck(xml, select): #select chooses m or p mode
    xmlevents = xml
    simevents = extractor(x_m, x_p, sr_m, sr_p, select)
    simevents=simevents[1:len(simevents)-1] #to get rid of the 0 and end event
    for i in range(len(simevents)):
       # e_pitch= ((simevents[i].pitch-xmlevents[i].pitch)/xmlevents[i].pitch)*100
        e_onset= ((simevents[i].onsetSec-xmlevents[i].onsetSec)/xmlevents[i].onsetSec)*100
      #  e_offset= ((simevents[i].offsetSec-xmlevents[i].offsetSec)/xmlevents[i].offsetSec)*100
        print(e_onset)

#errorcheck(xml_p, 1)
'''







def pitch_iterative(x, sr, n0, n1):
    thresh = 500
    x=x[n0:n1]
    X = abs(rfft(x))
    X=X[0:2100]
    for i in range(len(X)): #flatten
        if X[i]<=100:
            X[i]=0
    fakespec=np.zeros(len(X))
    current_max=max(X)
    print(current_max)
    if current_max<=thresh:
        print ("I'm done boiii")
    else:
        f0 = pitch_detect_HPS(x, sr, 4)
        for i in range(10):
            if i==0:
                pass
            elif i*f0<=2097:
                fakespec[(i*f0)-3:(i*f0)+3]=10000
        newspec=X-fakespec
        newspec=abs(newspec)
        b=[]



   # pitch_detect_HPS(x, sr, 4)


def onset_and_pitch(x, sr, mode): #mode=0->autocorellation, mode=1->HPS, mode=2->both, mode=3->Klapuri
    onset_boundaries = segmentation.onset_pad(segmentation.onset_samples_detect(x, sr), x, sr)
    print(onset_boundaries)
    #onsets=segmentation.onset_detect(x_m, x_p,  sr_m, sr_p)[0]
    frequencies=[]
    frequencies_hps=[]
    frequencies_klapuri=[]
    frequencies_pyfid=[]
    out = [frequencies, frequencies_hps]
    if mode==0: #autocorellation
        for i in range(len(onset_boundaries) - 1):
            n0 = onset_boundaries[i]
            n1 = onset_boundaries[i + 1]
            f0 = pitch_detect_autocorellation(x_m[n0:n1], sr_m)
            if f0 > 1250 or f0<75:
                f0 = 0
            frequencies.append(f0)
        return frequencies
    if mode==1: #Harmonic Product Spectrum:
        for i in range(len(onset_boundaries) - 1):
            n0 = onset_boundaries[i]
            n1 = onset_boundaries[i + 1]
            f0 = pitch_detect_HPS(x_m[n0:n1], sr_m, 4)
            if f0 > 1250 or f0<75:
                f0 = 0
            frequencies_hps.append(f0)
        return frequencies_hps
    if mode == 2: #both
        for i in range(len(onset_boundaries) - 1):
            n0 = onset_boundaries[i]
            n1 = onset_boundaries[i + 1]
            f0 = pitch_detect_autocorellation(x_m[n0:n1], sr_m)
            f1 = pitch_detect_HPS(x_m[n0:n1], sr_m, 4)
            if f0 > 1250 or f0<75:
                f0 = 0
            if f1 > 1250 or f1 < 75:
                f1 = 0
            frequencies.append(f0)
            frequencies_hps.append(f1)
        return out
    if mode ==3: #Klapuri
        for i in range(len(onset_boundaries) - 1):
            n0 = onset_boundaries[i]
            n1 = onset_boundaries[i + 1]
            freq_est = F0Estimate(max_poly=6)
            f0_estimates=freq_est.estimate_f0s(x_p[n0:n1], sr_p)
            frequencies_klapuri.append(f0_estimates)
        return frequencies_klapuri
    if mode ==4: #pypitch
        for i in range(len(onset_boundaries) - 1):
            n0 = onset_boundaries[i]
            n1 = onset_boundaries[i + 1]
            fundamentals, pitches, D, peaks, confidences, tracks = pyfid.ppitch(x[n0+128:n1-128])
            freq1=[]
            freq2=[]
            print("ITERATION NUMBER", i)
            print('FUNDAMENTALS', fundamentals)
            fundamentals=fundamentals.tolist()
            for j in range(len(fundamentals)-1):
                    freq1.append(fundamentals[j][0])
                    freq2.append(fundamentals[j][1])
            summand1=mean(freq1)
            summand2=mean(freq2)
            print('ONSET RANGE', i, 'WITH THE RANGE BEING:', onset_boundaries[i], 'to', onset_boundaries[i+1])
            print(summand1, summand2)



def cqtfind(x, sr):
    cqt = librosa.cqt(x, sr=sr, n_bins=300, bins_per_octave=bins_per_octave)
    log_cqt = librosa.amplitude_to_db(cqt)
    print("CQT Found. Dimensions:", cqt.shape)
    return log_cqt

def spectrogram(log_cqt, bins_per_octave, sr):
    librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note',
                             bins_per_octave=bins_per_octave)

#qt = librosa.cqt(x_p, sr=sr, n_bins=300, bins_per_octave=36)
#log_cqt = librosa.amplitude_to_db(np.abs(cqt))
#print("CQT Found. Dimensions:", cqt.shape)
#plt.figure()
#librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note',
    #                        bins_per_octave=36)

#onset_and_pitch(x_m, x_p, sr_m, sr_p, 4)
#print(frequencies)
#onsets=onset_detect(x_m, x_p, sr_m, sr_p)
#onsets_samples=onset_samples_detect(x_m, x_p, sr_m, sr_p)
#STFTspectre(x_m, sr_m, x_p, sr_p, onsets[0], onsets[1], frequencies[0], frequencies[1])

#x_test=x_m[38200: 76100]
#freq_from_HPS(x_test, sr_m, 4)

#D_m = librosa.stft(x_m)
#S_db_m = librosa.amplitude_to_db(np.abs(D_m), ref=np.max)

#onset_and_pitch(x_p, sr_p,4)

D = librosa.stft(x_p, n_fft=8192)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
ax.set(title='Using a logarithmic frequency axis')
fig.colorbar(img, ax=ax, format="%+2.f dB")
