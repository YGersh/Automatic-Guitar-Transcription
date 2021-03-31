from __future__ import division
import os
import argparse
import numpy as np
from scipy.signal import get_window
from utilities import nextpow2
import librosa, librosa.display
# set up command line argument structure
parser = argparse.ArgumentParser(description='Estimate the pitches in an audio file.')
parser.add_argument('-fin', '--filein', help='input file')
parser.add_argument('-fout', '--fileout', help='output file')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')


class F0Estimate:
    # sorted list of frequencies used to find the closest pitch
    # name and octave to the fundamental frequency estimate.
    # frequencies range from 65Hz to 2100Hz.
    frequencies = np.array([2 ** (n / 12.0) * 440 for n in range(-33, 28)])

    # sorted list of pitch name and octaves, which correspond to the
    # frequency elements in the frequency list.
    # for enharmonic notes, choose sharps instead of flats

    def __init__(self, **kwargs):
        # set maximum number of simultaneous notes
        self._max_poly = 6
        # set minimum fundamental frequency to detect
        self._min_f0 = 65
        # set maximum fundamental frequency to detect
        self._max_f0 = 2100
        # set analysis frame length
        self._frame_len_sec = 0.093 #Analysis frame length must be 46ms or 93ms
        self._window_func = 'hanning'
        # set the bin width of the estimated spectrum of the partial
        # of the detected fundamental
        self._partial_width = 10
        '''
        Derived parameters
        '''
        # these parameter values are from the 2006 paper by Klapuri
        if self._frame_len_sec == 0.046:
            self._alpha = 27
            self._beta = 320
            self._d = 1.0
        else:
            self._alpha = 52
            self._beta = 320
            self._d = 0.89

    def estimate_f0s(self, x, sr):
        # make x mono if stereo
        if x.ndim > 1:
            _, n_channels = x.shape
            x = x.sum(axis=1) / n_channels
        X = self._stft(x, sr)
        # Section 2.1 Spectrally whiten the signal to suppress timbral information
     #   Y = self._spectral_whitening(X, sr)
        Y=X
        # perform iterative estimation of the fundamental periods in the audio file
        f0_estimations = self._iterative_est(Y, sr)
        return f0_estimations

    def _stft(self, x, fs):
        '''
        Calculate short time fourier transform on signal that is
        hann windowed and zero-padded to twice its length.
        Hopsize = window length
        '''

        frame_len_samps = int(fs * self._frame_len_sec)
        win = get_window(self._window_func, frame_len_samps)

        # zero-pad to twice the length of the frame
        K = int(nextpow2(2 * frame_len_samps))
        X = np.array([np.fft.fft(win * x[i:i + frame_len_samps], K)
                      for i in range(0, len(x) - frame_len_samps, frame_len_samps)])

        return X

    def _spectral_whitening(self, X, sr, nu=0.33):
        '''
        Spectrally flatten ('whiten') the given input signal in the frequency domain,
        with the intention of supressing timbral information.
        PARAMETERS
        ----------
        X (T, K): frequency domain input signal with T frames and FFT of length K
        sr: sampling rate of the input signal
        nu (float): amount of spectral whitening
        '''

        T, K = X.shape
        nyquist_freq = sr / 2
        nyquist_bin = K >> 1

        # calculate centre frequencies c_b (Hz) of subbands on the critical-band scale
        # c_b = 229 * (10^[(b+1)/21.4]-1)
        # calculate one subband below and above the range to get the head and tail
        # frequencies of the triangle windows
        c = []  # centre frequencies of critical-bands
        b = 0  # critical band index
        while True:
            centre_freq = 229 * (10 ** ((b + 1) / 21.4) - 1)
            if centre_freq < nyquist_freq:
                c.append(centre_freq)
                b += 1
            else:
                break

        c = np.asarray(c)
        c_bins = np.asarray(np.floor(c * K / sr), np.int)

        # subband compression coefficients -> gamma (K/2,)
        gamma = np.zeros([T, nyquist_bin])

        # for each subband
        for b in range(1, len(c_bins) - 1):
            H = np.zeros(nyquist_bin)

            left = c_bins[b - 1]
            centre = c_bins[b]
            right = c_bins[b + 1]

            # construct the triangular power response for each subband
            H[left:centre + 1] = np.linspace(0, 1, centre - left + 1)
            H[centre:right + 1] = np.linspace(1, 0, right - centre + 1)

            # multiply by 2, since energy is symmetric about the nyquist rate
            gamma[:, centre] = np.sqrt((2 / K) * np.sum(H * (np.abs(X[:, :nyquist_bin]) ** 2), axis=1)) ** (nu - 1)

            # interpolate between the previous centre bin and the current centre bin
            # for each STFT frame
            for t in range(T):
                gamma[t, left:centre] = np.linspace(gamma[t, left], gamma[t, centre], centre - left)

        # calculate the whitened spectrum. Only need to store half the spectrum for analysis
        # since the bin energy is symmetric about the nyquist frequency
        Y = gamma * X[:, :nyquist_bin]

        return Y

    def _iterative_est(self, Y, sr):
        f0_estimations = []

        T = Y.shape[0]
        # for each STFT frame
        for t in range(T):
            # residual magnitude spectrum of the analysis frame
            Y_t_R = np.abs(Y[t, :])

            # fundamental frequency estimates for the current frame
            f0_frame_estimations = []

            # keep track of saliences of period estimates in this frame
            S = -1
            salience_hats = []

            # while there are fundamentals to estimate and the maximum number
            # of polyphony is not exceeded
            while len(salience_hats) < self._max_poly:
                tau_hat, salience_hat, Y_t_D = self._search_smax(Y_t_R, sr, tau_prec=0.5)
                salience_hats.append(salience_hat)

                f0_frame_estimations.append(sr / tau_hat)
                f0_estimations.append(f0_frame_estimations)

                cur_S = self._calc_S(salience_hats)
                if cur_S <= S:
                    break
                else:
                    # subtract the detected spectrum from the residual spectrum
                    Y_t_R -= self._d * Y_t_D
                    Y_t_R[Y_t_R < 0] = 0

                    S = cur_S

        return f0_estimations

    def _calc_S(self, salience_hats, gamma=0.7):
        '''
        Calculate a normalized sum of saliences to determine if searching
        for more fundamentals in the spectrum is necessary.
        '''

        j = len(salience_hats)
        S = sum(salience_hats) / (j ** gamma)

        return S

    def _search_smax(self, Y_t_R, fs, tau_prec=1.0):
        Q = 0  # index of the new block
        q_best = 0  # index of the best block

        tau_low = [round(fs / self._max_f0)]  # in samples/cycle
        tau_up = [round(fs / self._min_f0)]  # in samples/cycle
        smax = [0]

        while tau_up[q_best] - tau_low[q_best] > tau_prec:
            # split the best block and compute new limits
            Q += 1
            tau_low.append((tau_low[q_best] + tau_up[q_best]) / 2)
            tau_up.append(tau_up[q_best])
            tau_up[q_best] = tau_low[Q]

            # compute new saliences for the two block-halves
            for q in [q_best, Q]:
                salience, _ = self._calc_salience(Y_t_R, fs, tau_low[q], tau_up[q])
                if q == q_best:
                    smax[q_best] = salience
                else:
                    smax.append(salience)

            q_best = np.argmax(smax)

        # estimated fundamental period of the frame
        tau_hat = (tau_low[q_best] + tau_up[q_best]) / 2

        # calculate the spectrum of the detected fundamental period and harmonics
        salience_hat, harmonics = self._calc_salience(Y_t_R, fs, tau_low[q_best], tau_up[q_best])
        K = len(Y_t_R) << 1
        Y_t_D = self._calc_harmonic_spec(fs, K, harmonics)

        return tau_hat, salience_hat, Y_t_D

    def _calc_salience(self, Y_t_R, fs, tau_low, tau_up):
        salience = 0

        tau = (tau_low + tau_up) / 2
        delta_tau = tau_up - tau_low

        # calculate the number of harmonics under the nyquist frequency
        # the statement below is equivalent to floor((fs/2)/fo)
        num_harmonics = int(np.floor(tau / 2))

        # calculate all harmonic weights
        harmonics = np.arange(num_harmonics) + 1
        g = (fs / tau_low + self._alpha) / (harmonics * fs / tau_up + self._beta)

        # calculate lower and upper bounds of partial vicinity
        nyquist_bin = len(Y_t_R)
        K = nyquist_bin << 1
        lb_vicinity = K / (tau + delta_tau / 2)
        ub_vicinity = K / (tau - delta_tau / 2)

        # for each harmonic
        harmonics = []
        for m in range(1, num_harmonics + 1):
            harmonic_lb = round(m * lb_vicinity)
            harmonic_ub = min(round(m * ub_vicinity), nyquist_bin)
            harmonic_bin = np.argmax(Y_t_R[harmonic_lb - 1:harmonic_ub]) + harmonic_lb - 1
            harmonic_amp = Y_t_R[harmonic_bin]
            w_harmonic_amp = g[m - 1] * harmonic_amp

            # save the properties of this fundamental period and harmonics
            harmonics.append({'bin': harmonic_bin, 'amp': w_harmonic_amp})

            salience += w_harmonic_amp

        return salience, harmonics

    def _calc_harmonic_spec(self, fs, K, harmonics):
        nyquist_bin = K >> 1
        # initialize spectrum of detected harmonics
        Y_t_D = np.zeros(nyquist_bin)

        # calculate the partial spectrum for each harmonic
        # Klapuri PhD Thesis, page 62 and (Klapuri, 2006) Section 2.5
        # Even with these sources, the algorithm for estimating the
        # spectrum of the fundamental and partials is rather unclear.
        frame_len_samps = int(fs * self._frame_len_sec)
        win = get_window(self._window_func, frame_len_samps)
        window_spec = np.abs(np.fft.fft(win, K))
        partial_spectrum = np.hstack((window_spec[self._partial_width::-1],
                                      window_spec[1:self._partial_width + 1]))
        # normalize the spectrum
        partial_spectrum /= np.max(partial_spectrum)

        for h in harmonics:
            h_lb = max(0, h['bin'] - self._partial_width)
            h_ub = min(nyquist_bin - 1, h['bin'] + self._partial_width)

            # translate the spectrum of the window function to the position of the harmonic
            Y_t_D[h_lb:h_ub + 1] = h['amp'] * partial_spectrum[h_lb - h['bin'] + self._partial_width:h_ub - h[
                'bin'] + self._partial_width + 1]

        return Y_t_D

    def collapse_notes(self, notes):
        '''
        Collapse consecutive notes (notes that span more than
        one analysis frame).
        '''

        notes_c = []
        prev_frame = []
        for frame_n in notes:
            # remove identical notes
            if len(frame_n) > 1:
                n_set = set([n['pname'] + str(n['oct']) for n in frame_n])
                frame_n = [{'pname': n[:-1], 'oct': int(n[-1])} for n in n_set]

            # if polyphony is different, add to notes
            if len(frame_n) != len(prev_frame):
                notes_c.append(frame_n)
            elif not np.all([n1['pname'] == n2['pname'] and n1['oct'] == n2['oct']
                             for n1, n2 in zip(prev_frame, frame_n)]):
                notes_c.append(frame_n)

            prev_frame = frame_n

        return notes_c




#freq_est = F0Estimate(max_poly=6)
#f0_estimates, notes = freq_est.estimate_f0s(x, sr)
#notes_c = freq_est.collapse_notes(notes)
#req_est.write_mei(notes_c, output_path)


#filename_poly = 'DATASET/dataset3/audio/pathetique_poly.wav'
#x, sr =librosa.load(filename_poly)
#freq_est = F0Estimate(max_poly=6)
#f0_estimates = freq_est.estimate_f0s(x, sr)