import os

__author__ = 'timo'

"""
Compute and display a spectrogram.
Give WAV file as input
"""
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import bob
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def show_spectogram(wavfile):
    #wavfile = sys.argv[1]
    wavfile = 'data/01_Track_1.wav'


    (rate, signal) = scipy.io.wavfile.read(str(wavfile))

    signal = np.cast['float'](signal)
    signal = signal[:,0]

    # Parameters used to extract MFCC (These could be defined in a separate configuration file)
    wl = 20  # The window length in milliseconds
    ws = 10  # The window shift of the in milliseconds
    nf = 24  # The number of filter bands
    nceps = 19  # The number of cepstral coefficients
    fmin = 0.  # The minimal frequency of the filter bank
    fmax = 4000.  # The maximal frequency of the filter bank
    d_w = 2  # The delta value used to compute 1st and 2nd derivatives
    pre = 0.97  # The coefficient used for the pre-emphasis
    mel = True  # Tell whether MFCC or LFCC are extracted

    nc = 19  # Number of cepstral coefficients

    spectogrammer = bob.ap.Spectrogram(rate, wl, ws, nf, fmin, fmax, pre, mel)

    spectogram = spectogrammer(signal)


    plt.imshow(spectogram, aspect="auto", interpolation="none")
    plt.show()

    #sr, x = scipy.io.wavfile.read(wavfile)

    ## Parameters: 15ms step, 30ms window
    #nstep = int(sr * 0.015)
    #nwin = int(sr * 0.03)
    #nfft = nwin
    #
    #window = np.hamming(nwin)

    ## will take windows x[n1:n2].  generate
    ## and loop over n2 such that all frames
    ## fit within the waveform
    #nn = range(nwin, len(x), nstep)

    #X = np.zeros((len(nn), nfft / 2))

    #for i, n in enumerate(nn):
    #    xseg = x[n - nwin:n]
    #    z = np.fft.fft(window * xseg, nfft)
    #    X[i, :] = np.log(np.abs(z[:nfft / 2]))

    #plt.imshow(X.T, interpolation='nearest',
    #           origin='lower',
    #           aspect='auto')

    #plt.show()

def show_energy(wavfile):
    # Parameters used to extract MFCC (These could be defined in a separate configuration file)
    wl = 20  # The window length in milliseconds
    ws = 10  # The window shift of the in milliseconds

    (rate, signal) = scipy.io.wavfile.read(str(wavfile))

    signal = np.cast['float'](signal)
    signal = signal[:,0]

    energizer = bob.ap.Energy(rate, wl, ws)

    energy = energizer(signal)

    #plt.imshow(energy, aspect="auto", interpolation="none")
    plt.plot(energy)
   # plt.show()

def gen_log_energy_array(signal, rate):
    # Parameters used to extract MFCC (These could be defined in a separate configuration file)
    wl = 20  # The window length in milliseconds
    ws = 10  # The window shift of the in milliseconds

    energizer = bob.ap.Energy(rate, wl, ws)
    signal_float = np.cast['float'](signal)
    energy = energizer(signal_float)

    energy *= 10
    energy = np.log10(energy)

    energy[np.isneginf(energy)] = 0

    return energy

def find_low_points(array, window, trigger):
    logger.debug('Entering in find_low_points')

    min_points = {}
    search_min = False
    start = 0
    for i, value in enumerate(array):
        #print 'i = ' + str(i) + ' search_min = ' + str(search_min) + ' value = ' + str(value) + ' window = ' + str(window) + ' i-start = ' + str(i-start)
        if search_min and i-start < window:
            if value < min_value:
                min_value = value
                min_index = i

        else:
            if search_min:
                search_min = False
                min_points[min_index] = min_value
                #print 'Save min point'

            if value < trigger:
                search_min = True
                start = i
                min_value = value
                min_index = i

    logger.debug('Min points found: ' + str(min_points))
    return min_points

def mean_of_non_zero_values(array):
    logger.debug('Entering in mean_of_non_zero_values')
    #Calculate mean of values leaving out zero values
    value_sum = 0
    value_count = 0
    for value in array:
        if value != 0:
            value_sum += value
            value_count += 1

    mean = value_sum / float(value_count)

    logger.debug('values: ' + str(array))
    logger.debug('mean: ' + str(mean))
    return mean


def filter_out_large_values(array, max_value, max_pos_diff):
    logger.debug('Entering in filter_out_large_values')

    value_mean = mean_of_non_zero_values(array.viewvalues())

    print 'min_points_mean = ' + str(value_mean)

    remove_keys = []
    for key in array:
        value = array[key]
        if (value-value_mean)/max_value > max_pos_diff:
            remove_keys.append(key)

        print str((value-value_mean)/max_value*100)

    for key in remove_keys:
        logger.debug('Removing value [' + key + '] = ' + array[key])
        array.pop(key)

def find_silent_moments(signal, rate):
    logger.debug('Entering in find_silent_moments')
    #Calculate energy
    energy = gen_log_energy_array(signal, rate)
    #find and print  max value in energy array
    max_value = np.amax(energy)
#    print 'max energy (' + filename + ') = ' + str(max_value)


    min_win = 500
    min_trigger = 2.1

    logger.debug('Maximum energy = ' + str(max_value))

    min_points = find_low_points(energy, min_win, min_trigger)
    filter_out_large_values(min_points, max_value, 0.1)

    logger.debug('Found silent points: ' + str(min_points))
    return min_points.keys()

def split_signal_by_silence(signal, rate):
    logger.debug('Entering in split_signal_by_silence')
    silent_points = find_silent_moments(signal, rate)
    silent_points = [int(x * rate / 100) for x in silent_points]
    silent_points = sorted(silent_points)
    logger.debug('Split signal from points: ' + str(silent_points))
    signals = np.split(signal, silent_points)

    return signals

def gen_wav_filename(dest, name, number):

    wav_file = name + '_' + str(number) + '.wav'

    generated = os.path.join(dest, wav_file)
    logger.debug('Generated wav filename: ' + generated)
    return generated


def save_wavs(signals, rate, dest, name):
    i = 1
    for signal in signals:
        wav_file = gen_wav_filename(dest, name, i)
        logger.debug('Saving wav (' + wav_file + ')')
        scipy.io.wavfile.write(wav_file, rate, signal)
        i += 1

def load_wav_as_mono(wav_file):
    logger.debug('Entering in load_wav_as_mono')
    (rate, signal) = scipy.io.wavfile.read(str(wav_file))
    #signal = np.cast['float'](signal)
    signal = signal[:,0]
#    logger.debug('Loaded WAV file: signal size = ' + str(signal.shape) + ' rate = ' + str(rate) + ' length in ms = ')# + str(signal.shape[1] / rate))
    return signal, rate

def split_wav_by_silence(wav_file, dest_path):
    logger.debug('Splitting wav file (' + wav_file + ') to ' + dest_path)
    filename = os.path.basename(wav_file)
    filename_wo_ext = os.path.splitext(filename)[0]

    if not os.path.isfile(wav_file):
        return False

    signal, rate = load_wav_as_mono(wav_file)

    signals = split_signal_by_silence(signal, rate)

    save_wavs(signals, rate, dest_path, filename_wo_ext)

    logger.debug('Splitting done successfully')

    return True

def show_energy_plots(wavs):
    print 'wavs List:', str(wavs)
    num_plots = len(wavs)

    if num_plots < 1:
        print 'Files not given'
        return

    rows_map = [0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    cols_map = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]

    rows = rows_map[num_plots]
    cols = cols_map[num_plots]

    f, axarr = plt.subplots(rows, cols, sharex='col', sharey='row')

    plot = 0;

    for x in range(0, cols):
        for y in range(0, rows):
            if plot < num_plots:
                wavfile = os.path.abspath(str(wavs[plot]))

                if os.path.isfile(wavfile):
                    filename = os.path.basename(wavfile)

                    if cols == 1:
                        ax = axarr[y]

                    else:
                        ax = axarr[y, x]

                    #open wav file
                    rate, signal = load_wav_as_mono(wavfile)

                    #Calculate energy
                    energy = gen_log_energy_array(signal, rate)

                    #Plot energy array
                    ax.plot(energy)
                    ax.set_title(filename)

                    silent_points = find_silent_moments(signal, rate)

                    for point in silent_points:
                        ax.axvline(point, color='r')

                    #print min_points


                else:
                    print 'File not found: ' + wavfile
                plot += 1



    plt.show()


#show_energy_plots(sys.argv[1:])


wav_file = os.path.abspath(str(sys.argv[1]))
dest_path = os.path.abspath(str(sys.argv[2]))

split_wav_by_silence(wav_file, dest_path)


#(rate, signal) = scipy.io.wavfile.read(str(wav_file))
#(signal, rate) = load_wav_as_mono(wav_file)
#scipy.io.wavfile.write('/home/timo/temp/testi.wav', rate, signal)


