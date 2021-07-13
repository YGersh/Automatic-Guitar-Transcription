# Automatic-Guitar-Transcription
Capstone university project.

Input = real time guitar signal, output= tabulature of the played music.

Implements conversion of real time electric guitar audio to tablature. 

Important Libraries
------------------------
Numpy, Scipy, Matplotlib, SoundDevice, PeakUtils, Librosa, Matplotlib, Multiprocessing

Overview of the Program
------------------------
1\. Analog  signal enters ADC,  forwarded the computer as samples.

2\. Audio Input process polls a virtual audio device created by VoiceMeeter for samples using PortAudio.

3\. When a full, legitimate frame is collected, it is sent to segmentation.

4\. When Segmentation detects onsets (Spectral Flux Novelty), generate a slice containing the note.

5\. Send the slice to Pitch Detection, estimate the fundamental frequency (Autocorellation and Harmonic Product Spectrum).

6\. Send an object containing the segment, onset time and pitch to Tablature Generator (Heursitic), identify string & fret for the note.

7\. Forward the string & fret information to Tablature Display, which outputs a real time tab.
