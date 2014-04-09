import os

__author__ = 'timo'

"""
Compute and display a spectrogram.
Give WAV file as input
"""
import scipy.stats as stats
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import bob
import sys
import logging

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

FIGNUM = 0

matplotlib.rcParams.update({'font.size': 5})

def show_spectogram(wav_file):
    #wavfile = sys.argv[1]
    wav_file = 'data/01_Track_1.wav'

    (rate, signal) = scipy.io.wavfile.read(str(wav_file))

    signal = np.cast['float'](signal)
    signal = signal[:, 0]

    # Parameters used to extract MFCC (These could be defined in a separate configuration file)
    wl = 20  # The window length in milliseconds
    ws = 10  # The window shift of the in milliseconds
    nf = 24  # The number of filter bands
    #nceps = 19  # The number of cepstral coefficients
    fmin = 0.  # The minimal frequency of the filter bank
    fmax = 4000.  # The maximal frequency of the filter bank
    #d_w = 2  # The delta value used to compute 1st and 2nd derivatives
    pre = 0.97  # The coefficient used for the pre-emphasis
    mel = True  # Tell whether MFCC or LFCC are extracted

    #nc = 19  # Number of cepstral coefficients

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
    signal = signal[:, 0]

    energizer = bob.ap.Energy(rate, wl, ws)

    energy = energizer(signal)

    #plt.imshow(energy, aspect="auto", interpolation="none")
    plt.plot(energy)
    # plt.show()


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


def gen_log_energy_array(signal, rate, wl=10, ws=5):
    # Parameters used to extract MFCC (These could be defined in a separate configuration file)
    #wl = 20  # The window length in milliseconds
    #ws = 10  # The window shift of the in milliseconds

    energizer = bob.ap.Energy(rate, wl, ws)
    signal_float = np.cast['float'](signal)
    energy = energizer(signal_float)

    energy *= 10
    energy = np.log10(energy)

    energy[np.isneginf(energy)] = 0
    energy /= np.max(np.abs(energy), axis=0)
    #    energy = smooth(energy, 200)
    return energy


def find_low_points(array, window, trigger):
    logger.debug('Entering in find_low_points')

    min_points = {}
    search_min = False
    start = 0
    for i, value in enumerate(array):
        #print 'i = ' + str(i) + ' search_min = ' + str(search_min) + ' value = ' + str(value) + ' window = ' + str(window) + ' i-start = ' + str(i-start)
        if search_min and i - start < window:
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


def find_low_points(array, x, window, trigger):
    logger.debug('Entering in find_low_points')

    min_points = {}
    search_min = False
    start = 0
    for i, value in enumerate(array):
        #print 'i = ' + str(i) + ' search_min = ' + str(search_min) + ' value = ' + str(value) + ' window = ' + str(window) + ' i-start = ' + str(i-start)
        if search_min and i - start < window:
            if value < min_value:
                min_value = value
                min_index = i

        else:
            if search_min:
                search_min = False
                min_points[x[min_index]] = min_value
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

    if len(array) == 0:
        return 0

    #Calculate mean of values leaving out zero values
    value_sum = 0
    value_count = 0
    for value in array:
        if value != 0:
            value_sum += value
            value_count += 1

    if value_count > 0:
        mean = value_sum / float(value_count)
    else:
        mean = 0

    logger.debug('values: ' + str(array))
    logger.debug('mean: ' + str(mean))
    return mean


def is_valid_split_point(index, value, value_mean, value_max, signal_length, allowed_value_diff):
    #Invalid if split position is at the beginning or at the end of the signal
    if index <= 0 or index >= signal_length:
        return False

    #invalid if value is too large compared to normalized mean value
    if (value - value_mean) / value_max > allowed_value_diff:
        return False

    return True


def filter_out_large_values(array, max_value, signal_length, max_pos_diff):
    logger.debug('Entering in filter_out_large_values')

    value_mean = mean_of_non_zero_values(array.viewvalues())

    print 'min_points_mean = ' + str(value_mean)

    valid_split_points = {}
    invalid_split_points = {}

    for key in array:
        value = array[key]
        if is_valid_split_point(key, value, value_mean, max_value, signal_length, max_pos_diff):
            valid_split_points[key] = value
        else:
            invalid_split_points[key] = value

    return (valid_split_points, invalid_split_points)
    # remove_keys = []
    # for key in array:
    #     value = array[key]
    #     if (value-value_mean)/max_value > max_pos_diff:
    #         remove_keys.append(key)
    #
    #     print str((value-value_mean)/max_value*100)
    #
    # for key in remove_keys:
    #     logger.debug('Removing value [' + str(key) + '] = ' + str(array[key]))
    #     array.pop(key)


def find_silent_moments(energy, min_trigger=0.86, min_point_win=20, chunk_size=50):
    logger.debug('Entering in find_silent_moments')
    #Calculate energy

    #find and print  max value in energy array
    max_value = np.amax(energy)
    #    print 'max energy (' + filename + ') = ' + str(max_value)

    logger.debug('Maximum energy = ' + str(max_value))

    chunks = chunkyfy(energy, chunk_size)
    mean_filtered = stats.nanmean(chunks, axis=1)
    x = np.linspace(chunk_size / 2, energy.size - chunk_size / 2, mean_filtered.size)

    min_points = find_low_points(mean_filtered, x=x, window=min_point_win, trigger=min_trigger)

    (min_points, invalid_min_points) = filter_out_large_values(min_points, max_value, energy.size, 0.08)

    logger.debug('Found silent points: ' + str(min_points))
    return (min_points.keys(), invalid_min_points.keys())
    #return (min_points.keys())


def energy_is_silence(energy):
    return energy.mean() < 2.1


def mean_energy(signal, rate):
    e = gen_log_energy_array(signal, rate)
    return e.mean()


def validate_signal(signal, rate, min_energy=0.5, min_length=0.5):
    if signal.size == 0:
        logger.debug('Signal length is 0 -> signal is invalid')
        return False
    mean_e = mean_energy(signal, rate)
    signal_size = signal.size
    min_len = min_length*rate

    logger.debug('Validating signal: energy_mean: ' + str(mean_e) + ' len: ' + str(signal_size*rate))

    if signal.size < min_length*rate:
        logger.debug('Signal is too short (' + str(signal.size*rate) + ' < '+ str(min_length) +') -> Signal is invalid')
        return False

    logger.debug('Signal mean energy is ' + str(mean_e))

    if np.isnan(mean_e):
        logger.debug('Signal is empty ('+ str(mean_e) +')')
        return False

    if mean_e < min_energy:
        logger.debug('Signal mean energy ('+ str(mean_e) +') is under' + str(min_energy) + ' -> Signal is invalid')
        return False

    logger.debug('Signal is valid')

    return True


def split_signal_by_silence(signal, rate):
    logger.debug('Entering in split_signal_by_silence')
    energy = gen_log_energy_array(signal, rate)
    (silent_points_ms, invalid_points) = find_silent_moments(energy)
    silent_points = [int(x * rate / 100) for x in silent_points_ms]
    silent_points = sorted(silent_points)
    logger.debug('Split signal from points: ' + str(silent_points))
    signals = np.split(signal, silent_points)

    valid_signals = []
    invalid_signals = []
    for x in signals:
        if validate_signal(x, rate):
            valid_signals.append(x)
        else:
            invalid_signals.append(x)

    #signals[:] = [x for x in signals if not validate_signal(x, rate)]
    #signals[:] = [x for x in signals if not mean_energy(x, rate) < 2.1]

    return (valid_signals, invalid_signals)


def gen_wav_filename(dest, name, number):
    wav_file = name + '-split_' + str(number) + '.wav'

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
    signal = signal[:, 0]
    #    logger.debug('Loaded WAV file: signal size = ' + str(signal.shape) + ' rate = ' + str(rate) + ' length in ms = ')# + str(signal.shape[1] / rate))
    return signal, rate


def strip_filename(filename):
    filename = os.path.basename(filename)
    filename_wo_ext = os.path.splitext(filename)[0]
    return filename_wo_ext


def check_or_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    elif not os.path.isdir(path):
        logger.error('Not a directory: ' + path)
        return False
    return True


def split_wav_by_silence(wav_file, dest_path, invalid_dest):
    logger.debug('Splitting wav file (' + wav_file + ') to ' + dest_path)

    if not os.path.isfile(wav_file):
        return False

    signal, rate = load_wav_as_mono(wav_file)

    (signals, invalid_signals) = split_signal_by_silence(signal, rate)

    name = strip_filename(wav_file)

    logger.debug('Saving valid signals')
    if len(signals) > 0 and check_or_make_dir(dest_path):
        save_wavs(signals, rate, dest_path, name)

    #Save invalid parts that are not zero length
    invalid_signals[:] = [x for x in invalid_signals if x.size > 0]
    logger.debug('Saving invalid signals')
    if len(invalid_signals) > 0 and check_or_make_dir(invalid_dest):
        save_wavs(invalid_signals, rate, invalid_dest, name)

    logger.debug('Splitting done successfully')

    return True


def plot_wav_energy_with_splitpoints(wav_file, ax):
    filename = os.path.basename(wav_file)
    #open wav file
    signal, rate = load_wav_as_mono(wav_file)

    #Calculate energy
    energy = gen_log_energy_array(signal, rate)

    #Plot energy array
    ax.plot(energy)
    ax.plot(smooth(energy, 200), color='y')
    ax.set_title(filename)

    (silent_points, invalid_silent_points) = find_silent_moments(energy)

    for point in silent_points:
        ax.axvline(point, color='r')

    for point in invalid_silent_points:
        ax.axvline(point, color='g')


def plot_split_points(ax, split_points, invalid_split_points):
    for point in split_points:
        ax.axvline(point, color='r')

    for point in invalid_split_points:
        ax.axvline(point, color='g')


def plot_wav_with_splitpoints(ax, signal, split_points, invalid_split_points, name):
    ax.plot(signal)
    ax.set_title(name)
    plot_split_points(ax, split_points, invalid_split_points)


def split_list(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def chunkyfy(a, chunk_size=50):
    pad_size = math.ceil(float(a.size) / chunk_size) * chunk_size - a.size
    return np.append(a, np.zeros(pad_size) * np.NaN).reshape((-1, chunk_size))


def plot_wav(wav_file, fignum):
    f, axarr = plt.subplots(2, 1, False, False, False, num=fignum)
    (signal, rate) = load_wav_as_mono(wav_file)
    energy = gen_log_energy_array(signal, rate)

    filename = os.path.basename(wav_file)

    (energy_silent_points, energy_inv_silent_points) = find_silent_moments(energy)

    #energy_silent_points = find_silent_moments(energy)
    #signal_silent_points = [int(x * rate / 100) for x in energy_silent_points]
    #signal_inv_silent_points = [int(x * rate / 100) for x in energy_inv_silent_points]

    log_signal = signal * 10
    log_signal = np.log10(signal)

    #Plot log_signal
    #ax = axarr[0, 0]


    #  chunk_size = 44100
    #  pad_size = math.ceil(float(signal.size)/chunk_size)*chunk_size - signal.size
    #  signal = np.append(signal, np.zeros(pad_size)*np.NaN)
    #
    #  chunks = signal.reshape((-1, chunk_size))
    # # max_filtered = stats.nanmax(chunks, axis=1)
    # # min_filtered = stats.nanmin(chunks, axis=1)
    #  mean_filtered = stats.nanmean(chunks, axis=0)
    #  std_filtered = stats.nanstd(chunks, axis=0)
    #
    #  min_filtered = np.nanmin(chunks, axis=0)
    #  max_filtered = np.nanmax(chunks, axis=0)
    #std_filtered = std_filtered[~np.isnan(std_filtered)]

    #chunksize = 10000
    #numchunks = y.size // chunksize
    #ychunks = y[:chunksize*numchunks].reshape((-1, chunksize))
    #xchunks = x[:chunksize*numchunks].reshape((-1, chunksize))

    # Calculate the max, min, and means of chunksize-element chunks...
    #max_env = ychunks.max(axis=1)
    #min_env = ychunks.min(axis=1)
    #ycenters = ychunks.mean(axis=1)
    #xcenters = xchunks.mean(axis=1)

    # Now plot the bounds and the mean...

    #ax.fill_between(xcenters, min_env, max_env, color='gray',
    #                edgecolor='none', alpha=0.5)
    #    ax.plot(std_filtered, linewidth=0.4)


    #    ax.plot(log_signal)
    #    ax.plot(smooth(log_signal, 44000), color='y')
    #    ax.set_title(filename)
    #plot_split_points(ax, signal_silent_points, signal_inv_silent_points)

    #Plot signal array
    #    ax = axarr[1, 0]
    #    ax.plot(signal)
    #    ax.plot(max_filtered, linewidth=0.4)
    #    ax.plot(min_filtered, linewidth=0.4, color='g')
    #ax.plot(smooth(signal, 200), color='y')
    #    ax.set_title(filename)
    #plot_split_points(ax, signal_silent_points, signal_inv_silent_points)

    #Plot energy array

    chunk_size = 50
    chunks = chunkyfy(energy, chunk_size)

    mean_filtered = stats.nanmean(chunks, axis=1)
    min_filtered = np.nanmin(chunks, axis=1)
    max_filtered = np.nanmax(chunks, axis=1)
    std_filtered = stats.nanstd(chunks, axis=1)
    x = np.linspace(chunk_size / 2, energy.size - chunk_size / 2, min_filtered.size)

    ax = axarr[0, 0]
    ax.plot(energy, linewidth=0.4, color='gray')
    #ax.plot(smooth(energy, 200), color='y', linewidth=0.4)
    #ax.fill_between(x, min_filtered, max_filtered, color='gray',
    #            edgecolor='none', alpha=0.5)
    ax.plot(x, mean_filtered, color='b', linewidth=0.4)

    ax.set_title(filename)
    plot_split_points(ax, energy_silent_points, energy_inv_silent_points)

    ax = axarr[1, 0]
    plot_split_points(ax, energy_silent_points, energy_inv_silent_points)
    ax.plot(x, std_filtered, color='g', linewidth=0.4)

    return f


def plot_wavs(wavs, fignum, rows=4, cols=2):
    print 'wavs List:', str(wavs)
    num_plots = len(wavs)

    if num_plots < 1:
        print 'Files not given'
        return None

    f, axarr = plt.subplots(rows, cols, True, True, True, num=fignum)

    for x in range(0, cols):
        for y in range(0, rows):
            if len(wavs) > 0:
                wavfile = os.path.abspath(str(wavs.pop()))
            else:
                wavfile = None
                logger.debug('No files left')
                break

            if is_wav_file(wavfile):
                if cols > 1:
                    ax = axarr[y, x]
                else:
                    ax = axarr[y]

                plot_wav_energy_with_splitpoints(wavfile, ax)

        if wavfile is None:
            break

    return f


def recursive_list_wav_files(path, wavs=[]):
    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if is_wav_file(f_location):
            wavs.append(f_location)
        elif os.path.isdir(f_location):
            recursive_list_wav_files(f_location, wavs)
    return wavs


def is_wav_file(file):
    if not os.path.isfile(file):
        return False
    filename = os.path.basename(file)
    file_ext = os.path.splitext(filename)[1]
    return file_ext.lower() == '.wav'


def recursive_plot(path, pdf_file, img_path):
    wavs = recursive_list_wav_files(path)
    #wavs_split = split_list(wavs, 1)
    fignum = 0
    #figures = []
    #pp = PdfPages(pdf_file)
    for wav in wavs:
        #figure = plot_wavs(wav_split, fignum, 4, 2)
        figure = plot_wav(wav, fignum)
        filename = os.path.basename(wav)
        if figure != None:
            #pp.savefig(figure, dpi=200)
            image_file = os.path.join(img_path, str(filename) + '_fig' + str(fignum) + '.png')


            figure.savefig(image_file, dpi=200)
            #plt.show()
            #figures.append(figure)
        fignum += 1
        #plt.show()
        #if len(figures):

        #    pp.savefig(figures[0])
        #for fig in figures:
        #    pp.savefig(fig)
        #pp.close


def recursively_plot_wav_files(path, img_path, fignum=0):

    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if is_wav_file(f_location):
            logger.debug('Process file: ' + f_location)
            stripped_filename = strip_filename(f)
            #split_dir = os.path.join(img_path, stripped_filename)
            split_dir = img_path

            figure = plot_wav(unicode(f_location), fignum)
            if figure != None:
                check_or_make_dir(split_dir)
                #pp.savefig(figure, dpi=200)
                image_file = os.path.join(split_dir, str(f) + '_fig' + str(fignum) + '.png')
                logger.debug('Save plot to file: ' + image_file)
                figure.savefig(unicode(image_file), dpi=200)
                logger.debug('Plot saved')
                #plt.show()
                #figures.append(figure)
            fignum += 1

        elif os.path.isdir(f_location):
            logger.debug('Process location: ' + f_location)
            new_path = os.path.join(path, f)
            new_dest = os.path.join(img_path, f)
            recursively_plot_wav_files(new_path, new_dest, fignum)

        else:
            logger.info('Unknown file type: ' + str(f_location))
            continue
    return fignum


def recursively_split_wav_files(path, dest, invalid_dest):
    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if is_wav_file(f_location):
            logger.debug('Process file: ' + f_location)
            #stripped_filename = strip_filename(f)
            #split_dir = os.path.join(dest, stripped_filename)
            #invalid_split_dir = os.path.join(invalid_dest, stripped_filename)
            #split_wav_by_silence(f_location, split_dir, invalid_split_dir)
            split_wav_by_silence(f_location, dest, invalid_dest)

        elif os.path.isdir(f_location):
            logger.debug('Process location: ' + f_location)
            new_path = os.path.join(path, f)
            new_dest = os.path.join(dest, f)
            new_invalid_dest = os.path.join(invalid_dest, f)
            recursively_split_wav_files(new_path, new_dest, new_invalid_dest)

        else:
            logger.info('Unknown file type: ' + str(f_location))
            continue



#recursively_split_wav_files(orig_sound_path, sound_path, invalid_snd_path)
#recursively_plot_wav_files(orig_sound_path, split_plot_path)

#show_energy_plots(sys.argv[1:])


#wav_file = os.path.abspath(str(sys.argv[1]))
#src_path = os.path.abspath(str(sys.argv[1]))
#dest_path = os.path.abspath(str(sys.argv[2]))
#invalid_dest_path = os.path.abspath(str(sys.argv[3]))
#pdf_file = os.path.abspath(str(sys.argv[4]))
#img_file = os.path.abspath(str(sys.argv[5]))





#check_or_make_dir(img_file)


#recursively_split_wav_files(src_path, dest_path, invalid_dest_path)
#recursively_plot_wav_files(src_path, img_file)

#(rate, signal) = scipy.io.wavfile.read(str(wav_file))
#(signal, rate) = load_wav_as_mono(wav_file)
#scipy.io.wavfile.write('/home/timo/temp/testi.wav', rate, signal)

#(rate, signal) = scipy.io.wavfile.read(str('/Tavara/Ohjelmointi/BirDetect/birds/data/snd/train/Bird_Sounds_of_Europe_and_North-West-Africa/accipiter_nisus-split_4.wav'))


#valid = validate_signal(signal, rate)

#print 'valid=' + str(valid)