#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
#
# Copyright (C) 2014 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import numpy
from itertools import izip
import scipy.io.wavfile

def ensure_dir(dirname):
  """ Creates the directory dirname if it does not already exist,
      taking into account concurrent 'creation' on the grid.
      An exception is thrown if a file (rather than a directory) already 
      exists. """
  try:
    # Tries to create the directory
    os.makedirs(dirname)
  except OSError:
    # Check that the directory exists
    if os.path.isdir(dirname): pass
    else: raise


def read(filename):
  """Read audio file"""
  import scipy.io.wavfile
  rate, audio = scipy.io.wavfile.read(filename)
  
  # We consider there is only 1 channel in the audio file => data[0]
  data= numpy.cast['float'](audio) # pow(2,15) is used to get the same native format as for scipy.io.wavfile.read
  return [rate,data]



def normalize_features(feature):
  """Applies a unit mean and variance normalization to the features"""

  # Initializes variables
  n_samples = feature.shape[0]
  length = feature.shape[1]
  
  mean = numpy.ndarray((length,), 'float64')
  std = numpy.ndarray((length,), 'float64')

  mean.fill(0)
  std.fill(0)

  # Computes mean and variance
  for array in feature:
    x = array.astype('float64')
    mean += x
    std += (x ** 2)

  mean /= n_samples
  std /= n_samples
  std -= (mean ** 2)
  std = std ** 0.5 
  normalized = numpy.ndarray(shape=(n_samples,mean.shape[0]), dtype=numpy.float64)
    
  for i in range (0, n_samples):
    normalized[i,:] = (feature[i]-mean) / std 
  return normalized

def cosine_score(a, b, unit_vec=False):
    if len(a) != len(b):
        raise ValueError, "a and b must be same length"
    numerator = sum(tup[0] * tup[1] for tup in izip(a,b))
    if unit_vec:
      return numerator
    else:
      denoma = sum(avalue ** 2 for avalue in a)
      denomb = sum(bvalue ** 2 for bvalue in b)
      return numerator / (numpy.sqrt(denoma)*numpy.sqrt(denomb))

def is_wav_file(file):
    if not os.path.isfile(file):
        return False
    filename = os.path.basename(file)
    file_ext = os.path.splitext(filename)[1]
    return file_ext.lower() == '.wav'

def is_hdf5_file(file):
    if not os.path.isfile(file):
        return False
    filename = os.path.basename(file)
    file_ext = os.path.splitext(filename)[1]
    return file_ext.lower() == '.hdf5'

def file_has_extension(file, ext):
    if not os.path.isfile(file):
        return False
    filename = os.path.basename(file)
    file_ext = os.path.splitext(filename)[1]
    return file_ext.lower() == ext

def strip_filename(filename):
    filename = os.path.basename(filename)
    filename_wo_ext = os.path.splitext(filename)[0]
    return filename_wo_ext

def load_wav_as_mono(wav_file):
    #logger.debug('Entering in load_wav_as_mono')
    (rate, signal) = scipy.io.wavfile.read(str(wav_file))
    #signal = np.cast['float'](signal)
    if signal.ndim > 1:
        signal = signal[:, 0]
    #    logger.debug('Loaded WAV file: signal size = ' + str(signal.shape) + ' rate = ' + str(rate) + ' length in ms = ')# + str(signal.shape[1] / rate))
    return rate, signal
    

