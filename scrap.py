
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