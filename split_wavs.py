# coding=utf-8

import os

__author__ = 'Timo Mikkilä'

"""
Compute and display a spectrogram.
Give WAV file as input
"""
import scipy.stats as stats
import math
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import bob
import logging

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGNUM = 0

matplotlib.rcParams.update({'font.size': 5})

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('source', help='original sound path')
parser.add_argument('destination', help='split wavs')
parser.add_argument('--invalid_path', help='Invalid parts of wav files', default='invalis_splits')
parser.add_argument('--plots_path', help='Plots', default='plots')

args = parser.parse_args()

src_abs_path = os.path.abspath(args.source)
dest_abs_path = os.path.abspath(args.destination)
invalid_splits_path = os.path.abspath(args.invalid_path)
plots_path = os.path.abspath(args.plots_path)

logger.info('src_abs_path: ' + src_abs_path)
logger.info('dest_abs_path: ' + dest_abs_path)
logger.info('invalid_splits_path: ' + invalid_splits_path)
logger.info('plots_path: ' + plots_path)


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
    min_len = min_length * rate

    logger.debug('Validating signal: energy_mean: ' + str(mean_e) + ' len: ' + str(signal_size * rate))

    if signal.size < min_length * rate:
        logger.debug(
            'Signal is too short (' + str(signal.size * rate) + ' < ' + str(min_length) + ') -> Signal is invalid')
        return False

    logger.debug('Signal mean energy is ' + str(mean_e))

    if np.isnan(mean_e):
        logger.debug('Signal is empty (' + str(mean_e) + ')')
        return False

    if mean_e < min_energy:
        logger.debug('Signal mean energy (' + str(mean_e) + ') is under' + str(min_energy) + ' -> Signal is invalid')
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

    log_signal = signal * 10
    log_signal = np.log10(signal)


    chunk_size = 50
    chunks = chunkyfy(energy, chunk_size)

    mean_filtered = stats.nanmean(chunks, axis=1)
    min_filtered = np.nanmin(chunks, axis=1)
    max_filtered = np.nanmax(chunks, axis=1)
    std_filtered = stats.nanstd(chunks, axis=1)
    x = np.linspace(chunk_size / 2, energy.size - chunk_size / 2, min_filtered.size)

    ax = axarr[0, 0]
    ax.plot(energy, linewidth=0.4, color='gray')

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
    fignum = 0
    for wav in wavs:
        figure = plot_wav(wav, fignum)
        filename = os.path.basename(wav)
        if figure != None:
            image_file = os.path.join(img_path, str(filename) + '_fig' + str(fignum) + '.png')

            figure.savefig(image_file, dpi=200)

        fignum += 1



def recursively_plot_wav_files(path, img_path, fignum=0):
    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if is_wav_file(f_location):
            logger.debug('Process file: ' + f_location)
            stripped_filename = strip_filename(f)

            split_dir = img_path

            figure = plot_wav(unicode(f_location), fignum)
            if figure != None:
                check_or_make_dir(split_dir)
                image_file = os.path.join(split_dir, str(f) + '_fig' + str(fignum) + '.png')
                logger.debug('Save plot to file: ' + image_file)
                figure.savefig(unicode(image_file), dpi=200)
                logger.debug('Plot saved')
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


recursively_split_wav_files(src_abs_path, dest_abs_path, invalid_splits_path)
recursively_plot_wav_files(src_abs_path, plots_path)
