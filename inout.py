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
import mono


def parsexml (xmlfile): #load XML, create array of events
    tree = ET.parse(xmlfile) #load XML as ElementTree
    #root = tree.getroot() #Load Root for Metadata (unused rn)
    transcription = tree.find("transcription") #navigate to transcription section
    X_events=[]
    for i in range(len(transcription)): #create an array of Event Classes
        current_event = []
        current_event.append(transcription[i].find("pitch").text)
        current_event.append(transcription[i].find("onsetSec").text)
        current_event.append(transcription[i].find("offsetSec").text)
        current_event.append(transcription[i].find("fretNumber").text)
        current_event.append(transcription[i].find("stringNumber").text)
        current_event.append(transcription[i].find("excitationStyle").text)
        current_event.append(transcription[i].find("expressionStyle").text)
        X_events.append(Event(current_event))
    return X_events


