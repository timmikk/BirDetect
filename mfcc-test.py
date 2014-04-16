__author__ = 'timo'

import bob
import scipy.io.wavfile
import numpy

import matplotlib.pyplot as plt

#from matplotlib.pyplot import *
from scipy import *



# Feature extraction



def feature_extraction(audio_file):
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

    # We could also add 1st and 2nd derivatives by just activating their flags! See documentation for that

    #
    # TODO: read the audio file and getting the rate and the signal (hint: use utils.read)
    (rate, signal) = scipy.io.wavfile.read(str(audio_file))  #utils.read(audio_file)
    signal = numpy.cast['float'](signal)
    signal = signal[:,0]



    # TODO: extract MFCCs from the signal (hint: check the slides)
    #...
    ceps = bob.ap.Ceps(rate, wl, ws, nf, nc, fmin, fmax, d_w, pre, mel)

    #Signal should be float array

    mfcc = ceps(signal)

    # let's just normalize the features
    # This will reduce the effect of the channel
    mfcc = utils.normalize_features(mfcc)

    return mfcc

def show_MFCC(mfcc):
    """
    Show the MFCC as an image.
    """
    plt.imshow(mfcc.T, aspect="auto", interpolation="none")
    plt.title("MFCC features")
    plt.xlabel("Frame")
    plt.ylabel("Dimension")
    plt.show()

def split(wavfile):
    (rate, signal) = scipy.io.wavfile.read(str(wavfile))
    signal = np.cast['float'](signal)
    signal = signal[:,0]
    signal








