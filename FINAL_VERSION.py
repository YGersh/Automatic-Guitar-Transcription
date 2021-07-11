import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import math
import sounddevice as sd
import multiprocessing
import collections as c
import time
import peakutils as pk
from scipy.signal.windows import blackmanharris
from numpy.fft import rfft
from sortedcontainers import SortedDict
from itertools import islice
import operator
import csv

class noteobject:  # note info container
    def __init__(self, segment=[], onset_time=0.0, f0=0.0, estf0=0.0):
        self.segment = segment #contains samples belonging to the note
        self.onset_time = onset_time #onset time (in samples)
        self.f0 = f0 # fundamental frequency as determined by pitch detection
        self.estf0 = estf0 #fundamental frequency rounded to nearest legal note
        self.string = 0
        self.fret = 0

    def closest_note(self):
        '''
        Computes a legal note based on estimated f0
        '''
        key = self.f0
        frequency_map = SortedDict({
            82.41: 'E2',
            87.31: 'F2',
            92.5: ' F#2 ',
            98: 'G2',
            103.83: ' G#2',
            110: 'A2',
            116.54: 'A#2',
            123.47: 'B2',
            130.81: 'C3',
            138.59: 'C#3',
            146.83: 'D3',
            155.56: 'D#3',
            164.81: 'E3',
            174.61: 'F3',
            185: 'F#3',
            196: 'G3',
            207.65: 'G#3',
            220: 'A3',
            233.08: 'A#3',
            246.94: 'B3',
            261.63: 'C4',
            277.18: 'C#4',
            293.66: 'D4',
            311.13: 'D#4',
            329.63: 'E4',
            349.23: 'F4',
            369.99: 'F#4',
            392: 'G4',
            415.3: 'G#4',
            440: 'A4',
            466.16: 'A#4',
            493.88: 'B4',
            523.25: 'C5',
            554.37: 'C#5',
            587.33: 'D5',
            622.25: 'D#5',
            659.25: 'E5',
            698.46: 'F5',
            739.99: 'F#5',
            783.99: 'G5',
            830.61: 'G#5',
            880: 'A5',
            932.33: 'A#5',
            987.77: 'B5',
        })
        keys = list(islice(frequency_map.irange(minimum=key), 1))
        keys.extend(islice(frequency_map.irange(maximum=key, reverse=True), 1))
        self.estf0 = (min(keys, key=lambda k: abs(key - k)))



def sender(p1i):
    """
    Receives guitar signal from PortAudio, forwards to segmentation
    Must be as fast as possible
    """
    clock = time.clock()
    data = c.deque()  # deque is optimized for popping, allows us to save time
    def callback(indata, outdata, frames, time, status):
        if status:
            print("RECEIVED MESSAGE FROM DEVICE:", status)
        outdata[:] = indata #optional line, comment out to disable audio output
        data.append(indata)

    with sd.Stream(channels=1, callback=callback, samplerate=44100, blocksize=int(2205), dtype='float32',
                   latency='high'):
        # set high latency to avoid under/overflow
        # blocksize = 2205, i.e each buffer contains 2205 samples = 0.046sec
        while 1 == 1:
            try:
                a = data.popleft()
                if np.sum(a) > 0.1:
                    p1i.send(a)  # send here to avoid input overflow in the callback
            except IndexError:
                if time.clock() - clock > 5:
                    print("SENDER: Data Queue Empty")
                    clock = time.clock() #clock to avoid spamming "data queue empty" in case of silence


def segmenter(p1o, p2i):
    '''
    Receives frames, collects them in a running group, detects onsets, sends onset chunks to pitch detector
    :param p1o: Output end of audio input pipe
    :param p2i: Input end of pitch detection pipe
    :return: nothing
    '''

    # SETTINGS:
    width = 20  # how many frames to accumulate before onset detect
    added_width = 2 #how many frames to wait before re-running onset detection
    sr = 44100 #sampling rate
    song_rec = np.empty(0)  # this array will store entire history
    song_buf = np.empty(0) # while this one will store the current short-term group
    notes = []  # this is a list to use pop()
    onsets_old = []
    while True:
        if p1o.poll(None):
            msg = p1o.recv()
            song_buf = np.append(song_buf, np.ndarray.ravel(msg))
            if len(song_buf) == added_width * 2205: #if width satisfied append new frames to history
                song_rec = np.append(song_rec, (song_buf))
                song_buf = np.empty(0)
        if len(song_rec) >= width * 2205: #if total size satisfied attempt onset detection
            onsets = (librosa.onset.onset_detect(y=song_rec, sr=44100, hop_length=256, units='samples',
                                                 backtrack=True, pre_max=100, post_max=100, wait=10,
                                                 pre_avg=22050, post_avg=220, delta=0.15))
            if len(onsets_old) != len(onsets): #check if any new onsets have been found
                print("***********ONSET DETECTION**********")
                lendiff = len(onsets) - len(onsets_old)
                print("WE HAVE", lendiff, "NEW ELEMENTS")
                new_onsets = onsets[-lendiff:]
                onsets_old[:] = onsets
                print('DETECTED ONSETS AT:', onsets_old)
                print("OF WHICH NEW:", new_onsets)
            else:
                new_onsets = []
            if len(new_onsets) >= 1:  # if only 1 new onset
                segstart = onsets_old[onsets_old.index(new_onsets[0]) - 1]  # start at the onset before the new one
                segstop = new_onsets[0]
                seg = song_rec[segstart:segstop]
                if len(seg)>1:
                    current_note = noteobject(segment=seg, onset_time=librosa.samples_to_time(segstart, sr=44100))
                    print("MSG.SEG LENGTH", len(current_note.segment))
                    notes.append(current_note)
            if notes:
                note = notes.pop(0)
                p2i.send(note)




def parabolic(f, x):  # THIS IS A HELPER FUNCTION FOR HPS
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
    SOURCE: https://github.com/kwinkunks/notebooks/blob/master/parabolic.py
    """
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
    return xv, yv


def pitch_detector(p2o, p3i):
    def pitch_detect_autocorrelation(segment, sr, fmin=40, fmax=1400):
        r = librosa.autocorrelate(segment)
        i_min = sr / fmax #convert frequency bounds to time
        i_max = sr / fmin
        r[:int(i_min)] = 0 #assert frequency bounds
        r[int(i_max):] = 0
        i_peak = pk.indexes(r, thres=0.8, min_dist=5)[0] #use fancier peak detection for better accuracy
        i_interp = parabolic(r, i_peak)[0] #parabolic interpolation for the most precise result
        f0 = float(sr) / i_interp
        return f0

    def pitch_detect_HPS(x, sr, depth):
        windowed = x * blackmanharris(len(x))
        from pylab import subplot, plot, log, copy, show
        c = abs(rfft(windowed))
        maxharmms = depth
        freq = []
        for x in range(2, maxharmms):
            a = copy(c[::x])
            c = c[:len(a)]
            i = np.argmax(abs(c))
            true_i = parabolic(abs(c), i)[0]
            freq.append(sr * true_i / len(windowed))
            c *= a
        f0 = np.mean(freq)
        return int(f0)

    while True:
        if p2o.poll(None):
            msg = p2o.recv()
            size = len(msg.segment)
            # print("RECIEVED NOTE, len:", size)
            print("**********PITCH DETECTION**********")
            f0auto = pitch_detect_autocorrelation(msg.segment, sr=44100, fmin=20, fmax=1400)
            print("F-AUTO" , f0auto)
            f0hps  = pitch_detect_HPS(msg.segment, sr=44100, depth=4)
            print("F-HPS", f0hps)
            f0_out = (f0auto + f0hps) / 2
            sel_auto=False
            sel_hps=False
            sel_f0_out=False

            #HEURISTIC OUTPUT SELECTION LOGIC. LOGIC OUTLINED IN DECISION TREE IN PROJECT REPORT
            if f0auto > 1260 or f0auto < 70: #if f0auto garbled
                sel_auto=False
            if f0hps > 1260 or f0hps < 70: #if f0hps garbled
                sel_hps=False
            try: #try/except because of possible divisions by 0
                if math.isclose(f0auto, f0hps, abs_tol=32)==True: #if the values are close to each other output an average
                    sel_f0_out=True
                if math.isclose(f0auto/f0hps, 2, abs_tol=0.2)==True: #check for octave error in favor of hps
                    sel_hps=True
                elif math.isclose(f0hps/f0auto, 2, abs_tol=0.2)==True: #check for an octave error in favor of auto
                    sel_auto=True
            except:
                pass
            if f0auto-f0hps>f0hps:
                sel_hps=True
            elif f0hps-f0auto>f0auto:
                sel_auto=True
            if sel_auto+sel_hps+sel_f0_out >1: #if logic made mistake
                if sel_f0_out==True: #give prio to avg
                    msg.f0=f0_out
                else:
                    msg.f0=f0auto
            if sel_auto+sel_hps+sel_f0_out==0: #if no output was chosen
                sel_f0_out=True
            elif sel_auto==True:
                msg.f0=f0auto
            elif sel_hps==True:
                msg.f0=f0hps
            elif sel_f0_out==True:
                msg.f0=f0_out
            if msg.f0==0:
                msg.f0=min(f0auto, f0hps)
            print("FREQUENCY CHOSEN:", msg.f0)
            #msg.f0 = f0
            #print("COMPLETE NOTE OBJECT:")
            print('COMPLETE OBJECT HAS: \n ONSET TIME:', msg.onset_time, 'FREQUENCY:', msg.f0)
            p3i.send(msg)
    # print("DETECTED PITCH:", f0, 'Hz')


def tabulator(p3o, p4i):
    fret_map = np.array([  # mapping of frequency to string
        [329.63, 349.23, 369.99, 392, 415.3, 440, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46,
         739.99, 783.99, 830.61, 880, 932.33, 987.77, ],  # high E
        [246.94, 261.63, 277.18, 293.66, 311.13, 239.63, 349.99, 369.99, 392, 415.3, 440, 466.16, 493.88, 523.25,
         554.37, 587.33, 622.25, 659.25, 698.46, 739.99, ],
        [196, 207.65, 220, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392, 415.3, 440,
         466.16, 493.88, 523.25, 554.37, 587.33, ],
        [146.83, 155.56, 164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63,
         349.23, 369.99, 392, 415.3, 440, ],
        [110, 116.64, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94,
         261.63, 277.18, 293.66, 311.13, 329.63, ],
        [82.41, 87.31, 92.5, 98, 103.83, 110, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185, 196,
         207.65, 220, 233.08, 246.94, ]])  # low E

    def checkdist(p1, p2):
        '''
        Computes a weighted distance between two locations on the fretboard

        :param p1: Coordinates of first point
        :param p2: Coordinates of second point
        :return: Weighted distance between the inputs
        '''

        fret_weight = 1 #we care about fret distance more than string distance
        string_weight = 0.9
        if (p1[1] + p2[1]) - 24 > 0:
            playability_penalty = (((p1[1] + p2[1]) - 24) / 24) + 1  # apply penalty to high frets
        else: playability_penalty=1
        # [string, fret]
        stringdist = abs(p1[0] - p2[0])
        fretdist = abs(p1[1] - p2[1])
        dist = math.sqrt(((fretdist ** 2) * fret_weight + (stringdist ** 2) * string_weight)) * playability_penalty
        if p1[1]==0 or p2[1]==0: #if either note can be open give it priority
            dist=dist/2
        if p1[0]==p2[0]:
            dist=dist/2
        elif p1==p2:
            if p1[1] == 0 or p2[1] == 0:  # if either note can be open give it priority
                dist = dist / 2
        return dist

    def location_check(freq):
        '''
        Returns a set of coordinates where a given note could be placed
        :param freq: Fundamental frequency of the note
        :return: Array of possible locations
        '''
        possible_locations = []
        for x in range(0, fret_map.shape[0]):
            for y in range(0, fret_map.shape[1]):
                if fret_map[x, y] == freq:
                    #print("Input Frequency=", freq, "\nFound Frequency", fret_map[x, y], "\nAt location", [x, y])
                    possible_locations.append([x, y])
        #print("TOTAL POSSIBLE LOCATIONS:", possible_locations)
        return possible_locations

    def shortest_link(set1, set2):
        '''
        Chooses a pair of coordinates from two coordinate sets s.t. the weighted distance between them is minimized
        :param set1: Set of coordinates for note1
        :param set2: Set of coordinates for note2
        :return: Chosen coordinate pair (nested array)
        '''
        bestcoords = []
        bestdist = 99999
        if len(set1) > 1 and len(set2) > 1:  # if both lists have multiple candidates
            for i in range(len(set1)):
                for j in range(len(set2)):
                    distance = checkdist(set1[i], set2[j])
                    if distance < bestdist:
                        bestdist = distance
                        bestcoords = [set1[i], set2[j]]
        elif len(set1) == 1 and len(set2) > 1:
            for i in range(len(set2)):
                distance = checkdist(set1[0], set2[i])
                if distance < bestdist:
                    bestdist = distance
                    bestcoords = [set1[0], set2[i]]
        elif len(set1) == 1 and len(set2) == 1:  # special case
            bestcoords = [set1[0], set2[0]]
        return bestcoords

    # ACTIVE SECTION:
    history = []
    q = []
    tabulature = []
    popcount = 0
    notes = []
    while True:
        if p3o.poll(None):  # receive note object with onset time and
            msg = p3o.recv()

            msg.closest_note()  # estimate closest note to measured f0
            history.append(msg)
            q.append(msg)
        history.sort(key=operator.attrgetter('onset_time'))  # sort history to show notes in order of onset
        q.sort(key=operator.attrgetter('onset_time'))  # sort history to show notes in order of onset
        try:
            if len(q) >= 2 and popcount == 0:  # detect first iteration - 2 undetermined notes
                # obtain notes:
                note1 = q.pop(0)
                popcount += 1
                note2 = q.pop(0)
                # find optimal coordinate pair
                set1 = location_check(note1.estf0)
                set2 = location_check(note2.estf0)
                selection = shortest_link(set1, set2)
                # add this information to the note object
                note1.string = selection[0][0]
                note1.fret = selection[0][1]
                note2.string = selection[1][0]
                note2.fret = selection[1][1]
                # add these two notes to the tabulature
                p4i.send(note1)
                p4i.send(note2)
                tabulature.append(note1)
                tabulature.append(note2)
            elif q and popcount > 0:
                set1 = [[tabulature[-1].string, tabulature[-1].fret]]  # one of the notes is FIXED
                new_note = q.pop(0)
                set2 = location_check(new_note.estf0)
                selection = shortest_link(set1, set2)
                new_note.string = selection[1][0]
                new_note.fret = selection[1][1]
                # add the new note to tabulature:
                tabulature.append(new_note)
                p4i.send(new_note)
        except IndexError:  # indexerror raised if no notes in Queue
            print("TABULATOR WARNING: No notes in Queue")


def plotter(p4o):
    #OUTPUTS A REAL TIME SCROLLING TAB
    note_dist = 5  # distance between each note on the tabulator
    speed = 100
    transition_time_constant = 5  # used to make the notes appear to be sliding through time
    # PLOTTING SECTION:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylabel("String #")
    ax.set_xlabel("Time (s)")
    for i in range(1, 7):
        ax.axhline(y=i, color='black', linestyle='-', linewidth=3, alpha=0.6)
    ax.xaxis.set_visible(False)
    ax.set_title("Real-Time Tabulator")
    ax.set_xlim(-25, 0)
    pos = 0
    note_index = 1
    notes = []

    header = ['Position', 'Fret Number', 'String']
    f = open('tab.csv', 'w', newline='')
   # writer = csv.writer(f)
   # writer.writerow(header)
   # f.close()
    while True:
        notes.append(p4o.recv())
        note = notes[-1]
        new_row = [note_index, note.fret, note.string+1]
        #f = open('tab.csv', 'a', newline='')
       # writer = csv.writer(f)
       # writer.writerow(new_row)
       # f.close()
        ax.text(x=pos, y=note.string+1, s="|"+str(note.fret)+"|", fontsize=20, verticalalignment='center',
                horizontalalignment='center', clip_on=True, color='Blue', zorder=10)
        for i in range(1, note_dist*transition_time_constant + 1):
            ax.set_xlim(pos - 50 + i/transition_time_constant, pos + i/transition_time_constant)
            fig.canvas.draw()

            time.sleep(1 / (transition_time_constant * note_dist * speed))
        pos += note_dist
        note_index += 1


if __name__ == "__main__":

    # creating a pipe
    p1i, p1o = multiprocessing.Pipe()
    p2i, p2o = multiprocessing.Pipe()
    p3i, p3o = multiprocessing.Pipe()
    p4i, p4o = multiprocessing.Pipe()

    # creating new processes
    p1 = multiprocessing.Process(target=sender, args=(p1i,))
    p2 = multiprocessing.Process(target=segmenter, args=(p1o, p2i))
    p3 = multiprocessing.Process(target=pitch_detector, args=(p2o, p3i))
    p4 = multiprocessing.Process(target=tabulator, args=(p3o, p4i))
    p5 = multiprocessing.Process(target=plotter, args=(p4o,))

    # running processes
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    # wait until processes finish
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    while True: #prevent program exit
        pass
