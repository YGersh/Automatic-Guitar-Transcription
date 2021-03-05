from _future_ import division
import matplotlib
import seaborn
import numpy as np
import numpy.fft as fft
import scipy, scipy.fftpack, IPython.display as ipd, matplotlib.pyplot as plt
import librosa, librosa.display
import simpleaudio as sa
import xml.etree.cElementTree as ET
import peakutils as pk
import copy
from numpy import polyfit, arange
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal.windows import blackmanharris
#Import wall includes libraries that are not used. Will be cleaned up once code is complete.



class Event: #declare in format pitch, onsetSec, offsetSec, fretNumber, stringNumber, excitationStyle, expressionStyle
    def _init_(self, params) :
        self.pitch = float(params[0])
        self.onsetSec = float(params[1])
        self.offsetSec = float(params[2])
        self.fretNumber = float(params[3])
        self.stringNumber = float(params[4])
        self.excitationStyle = str(params[5])
        self.expressionStyle = str(params[6])
	def notecompare(self, target): #move error checking code into the class
		pass



filename_mono = 'DATASET/dataset3/audio/pathetique_mono.wav'
xmlfile_mono = 'DATASET/dataset3/annotation/pathetique_mono.xml'
filename_poly = 'DATASET/dataset3/audio/pathetique_poly.wav'
xmlfile_poly = 'DATASET/dataset3/annotation/pathetique_poly.xml'



hop_length=100 #give AI control of this parameter


x_m, sr_m = librosa.load(filename_mono)
x_p, sr_p=librosa.load(filename_poly)

def parsexml (xmlfile): #load XML annotation, create array of events
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




def STFTspectre(x_m, sr_m, x_p, sr_p, onsets_m, onsets_p, frequencies_auto, frequencies_hps): #display spectrogram of monophonic & polyphonic versions of signal, temporary func for parralelized workflow
    D_m = librosa.stft(x_m)
    S_db_m = librosa.amplitude_to_db(np.abs(D_m), ref=np.max)
    D_p = librosa.stft(x_m)
    S_db_p = librosa.amplitude_to_db(np.abs(D_p), ref=np.max)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    img1 = librosa.display.specshow(S_db_m, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='Onset  & Pitch Detection, Autocorellation')
    onset_boundaries = onset_samples_detect(x_m, x_p, sr_m, sr_p)[0]
    onset_boundaries = onset_pad(onset_boundaries, onset_samples_detect(x_m, x_p, sr_m, sr_p)[1], x_m, x_p, sr_m, sr_p)[0]
    segments = []
    for i in range(len(onset_boundaries)-1):
        segments.append(librosa.samples_to_time([onset_boundaries[i], onset_boundaries[i+1]], sr=sr_m))
    if len(onsets_m)>0:
        for i in range(len(onsets_m)-1):
           ax[0].axvline(onsets_m[i], color='r',)
    for i in range(len(frequencies_auto)):
        x_coords=segments[i]
        y_coords=[frequencies_auto[i], frequencies_auto[i]]
        ax[0].plot(x_coords, y_coords, color='b', linewidth=3)
    img2 = librosa.display.specshow(S_db_p, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set(title='Onset & Pitch Detection, HPS')
    if len(onsets_p)>0:
        for i in range(len(onsets_m)):
           ax[1].axvline(onsets_m[i], color='r',)
    for i in range(len(frequencies_hps)):
        x_coords=segments[i]
        y_coords=[frequencies_hps[i], frequencies_hps[i]]
        ax[1].plot(x_coords, y_coords, color='g', linewidth=3)
    for ax_i in ax:
        ax_i.label_outer()




def novelty(x_m, sr_m, x_p, sr_p, toplot):  # generates a novelty function, plots if 3rd arg=1
    onset_env_m = librosa.onset.onset_strength(x_m, sr=sr_m, hop_length=100)
    onset_env_p = librosa.onset.onset_strength(x_p, sr=sr_p, hop_length=100)
    if toplot == 1:
        plt.subplot(211)
        plt.plot(onset_env_m)
        plt.title('Novelty Function (Monophonic)')
        plt.xlabel("time")
        plt.ylabel("novelty level")
        plt.tight_layout()
        plt.subplot(212)
        plt.plot(onset_env_p)
        plt.title('Novelty Function (Polyphonic)')
        plt.xlabel("time")
        plt.ylabel("novelty level")
        plt.tight_layout()
      #  ax[1].set(title='Novelty Function (Mono)')

    return onset_env_m

# novelty(x_m, sr_m, x_p, sr_p, 1)



def onset_samples_detect(x_m, x_p, sr_m, sr_p):  # detect samples of onset BACKTRACKING ENABLED
    onset_samples_m = librosa.onset.onset_detect(x_m, sr=sr_m, units='samples', hop_length=hop_length, backtrack=True,
                                               pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=0)
    onset_samples_p = librosa.onset.onset_detect(x_p, sr=sr_p, units='samples', hop_length=hop_length, backtrack=True,
                                                 pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=0)
    onset_samples = [onset_samples_m, onset_samples_p]
    return onset_samples

def onset_pad(onset_samples_m, onset_samples_p, x_m, x_p, sr_m, sr_p):  # include stard and end of signal in onsets
    onset_bounds_m = np.concatenate(
        [[0], onset_samples_m, [len(x_m)]])  # include start and end of signal as onsets of notes
    onset_bounds_p = np.concatenate(
        [[0], onset_samples_p, [len(x_p)]])
    onset_bounds=[onset_bounds_m, onset_bounds_p]
    return onset_bounds

def onset_to_time(onset_bounds, sr):
    onset_times = librosa.samples_to_time(onset_bounds, sr=sr)
    return onset_times

def onset_detect(x_m, x_p, sr_m, sr_p):  # single func to call to get time onsets
    onsets_m=onset_to_time(onset_pad(onset_samples_detect(x_m,x_p, sr_m, sr_p)[0],onset_samples_detect(x_m,x_p, sr_m, sr_p)[1], x_m, x_p, sr_m, sr_p)[0], sr_m)
    onsets_p=onset_to_time(onset_pad(onset_samples_detect(x_m,x_p, sr_m, sr_p)[0],onset_samples_detect(x_m,x_p, sr_m, sr_p)[1], x_m, x_p, sr_m, sr_p)[1], sr_p)
    print("MONO:", onsets_m)
    print("POLY", onsets_p)
    onsets=[onsets_m, onsets_p]
    return onsets

xml_m=parsexml(xmlfile_mono)
xml_p=parsexml(xmlfile_poly)

#onsets=onset_detect(x_m, x_p, sr_m, sr_p)
#STFTspectre(x_m, sr_m, x_p, sr_p, onsets[0], onsets[1])

def extractor(x_m, x_p, sr_m, sr_p, select): #make event struct w extracted params
    onsets = onset_detect(x_m, x_p, sr_m, sr_p)[select]
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



def errorcheck(xml, select): #select chooses m or p mode
    xmlevents = xml
    simevents = extractor(x_m, x_p, sr_m, sr_p, select)
    simevents=simevents[1:len(simevents)-1] #to get rid of the 0 and end event
    for i in range(len(simevents)):
       # e_pitch= ((simevents[i].pitch-xmlevents[i].pitch)/xmlevents[i].pitch)*100
        e_onset= ((simevents[i].onsetSec-xmlevents[i].onsetSec)/xmlevents[i].onsetSec)*100
      #  e_offset= ((simevents[i].offsetSec-xmlevents[i].offsetSec)/xmlevents[i].offsetSec)*100
        print(e_onset)




def pitch_detect_autocorellation(segment, sr, fmin=40, fmax=1400): #pitch detect using autocorellation UNOPTIMIZED IMPLEMENTATION  - CHANGE. 
    r = librosa.autocorrelate(segment)
    i_min = sr / fmax
    i_max = sr / fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
   # i = r.argmax()
   #  f0 = float(sr) / i
    i_peak = pk.indexes(r, thres=0.8, min_dist=5)[0] #use peakutils for more robust peak detection, since numpy threw error
    i_interp = parabolic(r, i_peak)[0]
    f0=float(sr)/i_interp
    return f0


def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def onset_and_pitch(x_m, x_p,  sr_m, sr_p, mode): #mode=0->autocorellation, mode=1->HPS
    onset_boundaries = onset_samples_detect(x_m, x_p,  sr_m, sr_p)[0]
    onset_boundaries = onset_pad(onset_boundaries, onset_samples_detect(x_m, x_p,  sr_m, sr_p)[1],x_m, x_p, sr_m, sr_p)[0]
    onsets=onset_detect(x_m, x_p,  sr_m, sr_p)[0]
    frequencies=[]
    frequencies_hps=[]
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


def parabolic(f, x): #HELPER FUNCTION FOR HPS
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)
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
        true_i = parabolic(abs(c), i)[0]
        freq.append(sr*true_i/len(windowed))
        c *= a
    f0=np.mean(freq)
    return f0





frequencies=onset_and_pitch(x_m, x_p, sr_m, sr_p, 2)
onsets=onset_detect(x_m, x_p, sr_m, sr_p)
#onsets_samples=onset_samples_detect(x_m, x_p, sr_m, sr_p)
STFTspectre(x_m, sr_m, x_p, sr_p, onsets[0], onsets[1], frequencies[0], frequencies[1])

#x_test=x_m[38200: 76100]
#freq_from_HPS(x_test, sr_m, 4)
