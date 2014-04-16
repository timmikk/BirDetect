# coding=utf-8
import bob
import numpy
import os
import logging
from collections import defaultdict

import utils

__author__ = 'Timo Mikkilä'

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

current_dir = os.getcwd()

# base_path = ''
# snd_path = os.path.join(base_path, 'snd')
# train_sounds_path = os.path.join(snd_path, 'train')
# eval_sounds_path = os.path.join(snd_path, 'eval')
#
# analyze_path = os.path.join(base_path, 'analyze')
#
# train_features_path = os.path.join(analyze_path, 'train')
# eval_features_path = os.path.join(analyze_path, 'eval')
#
# mfcc_path = os.path.join(analyze_path, 'mfcc')
# kmeans_file = os.path.join(analyze_path, 'kmeans.hdf5')
# ubm_file = os.path.join(analyze_path, 'ubm.hdf5')
# gmm_stats_path = os.path.join(analyze_path, 'gmm_stats')
# tv_file = os.path.join(analyze_path, 'tv.hdf5')
# ivec_dir = os.path.join(analyze_path, 'ivectors')

# num_gauss = 16
#
# ubm_convergence_threshold = 1e-4
# ubm_max_iterations = 10

# Feature extraction
def feature_extraction(audio_file, wl=20, ws=10, nf=24, nceps=19, fmin=0., fmax=4000., d_w=2, pre=0.97, mel=True):
  # Parameters used to extract MFCC (These can be defined in a separate configuration file)
  # wl = 20 # The window length in milliseconds
  # ws = 10 # The window shift of the in milliseconds
  # nf = 24 # The number of filter bands
  # nceps = 19 # The number of cepstral coefficients
  # fmin = 0. # The minimal frequency of the filter bank
  # fmax = 4000. # The maximal frequency of the filter bank
  # d_w = 2 # The delta value used to compute 1st and 2nd derivatives
  # pre = 0.97 # The coefficient used for the pre-emphasis
  # mel = True # Tell whether MFCC or LFCC are extracted

  # We could also add 1st and 2nd derivatives by activating their flags!

  # read the audio file
  #(rate, signal) = utils.read(audio_file)
  (rate, signal) = utils.load_wav_as_mono(audio_file)

  # extract MFCCsu
  ceps = bob.ap.Ceps(rate, wl, ws, nf, nceps, fmin, fmax, d_w, pre, mel)

  #Convert signal to float array
  signal = numpy.cast['float'](signal)
  mfcc = ceps(signal)

  # let's just normalize them using this helper function
  # This will reduce the effect of the channel
  mfcc = utils.normalize_features(mfcc)

  return mfcc


def recursively_extract_features(path, dest_dir, wl=20, ws=10, nf=24, nceps=19, fmin=0., fmax=4000., d_w=2, pre = 0.97, mel=True):
    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if utils.is_wav_file(f_location):
            logger.debug('Extract features: ' + f_location)
            stripped_filename = utils.strip_filename(f)
            dest_file = os.path.join(dest_dir, stripped_filename)
            dest_file += '.hdf5'

            mfcc = feature_extraction(f_location, wl, ws, nf, nceps, fmin, fmax, d_w, pre, mel)
            utils.ensure_dir(dest_dir)

            bob.io.save(mfcc, dest_file)
            logger.debug('savin mfcc to ' + dest_file)


        elif os.path.isdir(f_location):
            logger.debug('Extract features: ' + f_location)
            new_path = os.path.join(path, f)
            new_dest_dir = os.path.join(dest_dir, f)
            recursively_extract_features(new_path, new_dest_dir)

        else:
            logger.info('Unknown file type: ' + str(f_location))
            continue


def recursive_load(path, file_validator, loader):
    data = []

    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if file_validator(f_location):
            logger.debug('Loading: ' + f_location)
            feature = loader(f_location)
            data.append(feature)


        elif os.path.isdir(f_location):
            logger.debug('Load dir: ' + f_location)
            new_path = os.path.join(path, f)
            data += recursive_load(new_path, file_validator, loader)

        else:
            logger.info('Unknown file type: ' + str(f_location))
            continue

    return data

def test_array_for_nan(array):
    for v in array:
        if type(v) is numpy.ndarray:
            test_array_for_nan(v)
        else:
            if numpy.isnan(v).any():
                print 'Contains NAN: ' + str(v)

            if numpy.isinf(v).any():
               print 'Contains INF: ' + str(v)


def recursive_test_for_nan(path):

    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if utils.is_hdf5_file(f_location):
            #logger.debug('Loading: ' + f_location)
            feature = bob.io.load(f_location)

            if numpy.isnan(feature).any():
                print 'Contains NAN: ' + f_location

            if numpy.isinf(feature).any():
                print 'Contains INF: ' + f_location

            #for dim in feature:
            #    for v in dim:
            #        if numpy.isnan(v):
            #            print 'Contains NAN: ' + f_location


        elif os.path.isdir(f_location):
            #logger.debug('Load dir: ' + f_location)
            new_path = os.path.join(path, f)
            recursive_test_for_nan(new_path)

        else:
            logger.info('Unknown file type: ' + str(f_location))
            continue

def recursive_test_wavs_for_nan(path):

    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        if utils.is_wav_file(f_location):
            #logger.debug('Loading: ' + f_location)
            rate, feature = utils.load_wav_as_mono(f_location)


            if numpy.isnan(feature).any():
                print 'Contains NAN: ' + f_location

            if numpy.isinf(feature).any():
                print 'Contains INF: ' + f_location

            #for dim in feature:
            #    for v in dim:
            #        if numpy.isnan(v):
            #            print 'Contains NAN: ' + f_location


        elif os.path.isdir(f_location):
            #logger.debug('Load dir: ' + f_location)
            new_path = os.path.join(path, f)
            recursive_test_wavs_for_nan(new_path)

        else:
            logger.info('Unknown file type: ' + str(f_location))
            continue

def load_mfcc_files(files):
    features = []
    for f in files:
        if not utils.is_hdf5_file(f):
            logger.error('Wrong file format: ' + f)
        features.append(bob.io.load(f))

    return features

def recursive_load_mfcc_files(path):
    return recursive_load(path, utils.is_hdf5_file, bob.io.load)

    # train_features = []
    #
    # for f in os.listdir(path):
    #     f_location = os.path.join(path, f)
    #     if utils.is_mfcc_file(f_location):
    #         logger.debug('Load mfcc: ' + f_location)
    #         feature = bob.io.load(f_location)
    #         train_features.append(feature)
    #
    #
    #     elif os.path.isdir(f_location):
    #         logger.debug('Load mfcc: ' + f_location)
    #         new_path = os.path.join(path, f)
    #         recursive_load_mfcc_files(new_path)
    #
    #     else:
    #         logger.info('Unknown file type: ' + str(f_location))
    #         continue
    #
    # return train_features

def train_kmeans_machine(train_features, num_gauss, dim):

    #dim = train_features.shape[1]
    kmeans_trainer = bob.trainer.KMeansTrainer()
    kmeans = bob.machine.KMeansMachine(num_gauss, dim)
    kmeans_trainer.train(kmeans, train_features)
    return kmeans


def train_ubm_gmm_with_features(kmeans, train_features, num_gauss, dim, convergence_threshold = 1e-4, max_iterations=10):
    #dim = train_features.shape[1]

    #Create universal background model
    ubm = bob.machine.GMMMachine(num_gauss, dim)

    # Initialize the means of the Gaussians of the GMM with the means of the k-means
    ubm.means = kmeans.means

    # update means/variances/weights at each iteration
    ubm_trainer = bob.trainer.ML_GMMTrainer(True, True, True)
    ubm_trainer.convergence_threshold = convergence_threshold
    ubm_trainer.max_iterations = max_iterations
    ubm_trainer.train(ubm, train_features)

    return ubm

def recursive_find_all_files(path, extension):
    logger.debug('Recursively list all files from ' + path + ' with extension ' + extension)
    files = []
    for f in os.listdir(path):
        f_location = os.path.join(path, f)
        #logger.debug('compute_gmm_sufficient_statistics: ' + f_location)

        if utils.file_has_extension(f_location, extension):
            files.append(f_location)

        elif os.path.isdir(f_location):
            child_path = os.path.join(path, f)
            files += recursive_find_all_files(child_path, extension)

        else:
            logger.debug('Wrong filetype: ' + str(f_location))
            continue

    logger.debug('Found ' + str(len(files)) + ' files')
    return files

def compute_gmm_sufficient_statistics(ubm, src_dir, dest_dir):
    for f in os.listdir(src_dir):
        f_location = os.path.join(src_dir, f)
        logger.debug('compute_gmm_sufficient_statistics: ' + f_location)

        if utils.is_hdf5_file(f_location):

            stripped_filename = utils.strip_filename(f)
            dest_file = os.path.join(dest_dir, stripped_filename)
            dest_file += '.hdf5'

            feature = bob.io.load(f_location)
            gmm_stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
            ubm.acc_statistics(feature, gmm_stats)

            utils.ensure_dir(dest_dir)

            gmm_stats.save(bob.io.HDF5File(dest_file, 'w'))
            logger.debug( 'savin gmm_stats to ' + dest_file)


        elif os.path.isdir(f_location):
            new_path = os.path.join(src_dir, f)
            new_dest_dir = os.path.join(dest_dir, f)
            compute_gmm_sufficient_statistics(ubm, new_path, new_dest_dir)

        else:
            logger.info('Unknown file type: ' + str(f_location))
            continue

def load_gmm_stats_file(file):
    return bob.machine.GMMStats(bob.io.HDF5File(file))

def load_gmm_machine_file(file):
    return bob.machine.GMMMachine(bob.io.HDF5File(file))

def recursive_load_gmm_stats(gmm_stats_path):
    return recursive_load(gmm_stats_path, utils.is_hdf5_file, load_gmm_stats_file)

def recursive_load_gmm_machines(gmm_machines_path):
    return recursive_load(gmm_machines_path, utils.is_hdf5_file, load_gmm_machine_file)

def gen_ivec_machine(gmm_train_stats, ubm, iv_dim=None, variance_treshold=1e-5, max_iterations=10, update_sigma=True):
    if iv_dim is None:
        iv_dim = len(gmm_train_stats)
    #Creating the i-vector machine
    ivec_machine = bob.machine.IVectorMachine(ubm, iv_dim)
    ivec_machine.variance_threshold = variance_treshold

    #Train the TV matrix
    ivec_trainer = bob.trainer.IVectorTrainer(update_sigma=update_sigma, max_iterations=max_iterations)
    ivec_trainer.train(ivec_machine, gmm_train_stats)

    return ivec_machine

def gmm_stats_to_ivec(gmm_stats_file, ivec_file, ivec_machine):
    # load the GMM stats file
    gmm_stats = bob.machine.GMMStats(bob.io.HDF5File(gmm_stats_file))

    # extract i-vector
    ivec = ivec_machine.forward(gmm_stats)

    # save them!
    bob.io.save(ivec, ivec_file)


def extract_i_vectors(gmm_stats_path, ivec_dir, ivec_machine):
    for f in os.listdir(gmm_stats_path):
            f_location = os.path.join(gmm_stats_path, f)
            logger.debug('extract_i_vectors: ' + f_location)

            if utils.is_hdf5_file(f_location):

                stripped_filename = utils.strip_filename(f)
                dest_file = os.path.join(ivec_dir, stripped_filename)
                dest_file += '.hdf5'

                utils.ensure_dir(ivec_dir)

                gmm_stats_to_ivec(f_location, dest_file, ivec_machine)

                logger.debug('saving ivec ' + dest_file)


            elif os.path.isdir(f_location):
                new_path = os.path.join(gmm_stats_path, f)
                new_dest_dir = os.path.join(ivec_dir, f)
                extract_i_vectors(new_path, new_dest_dir, ivec_machine)

            else:
                logger.info('Unknown file type: ' + str(f_location))
                continue

def map_gmm_machine_analysis(features_path, map_gmm_dir, map_gmm_machine):
    for f in os.listdir(features_path):
            f_location = os.path.join(features_path, f)
            logger.debug('running map gmm machine for: ' + f_location)

            if utils.is_hdf5_file(f_location):

                stripped_filename = utils.strip_filename(f)
                dest_file = os.path.join(map_gmm_dir, stripped_filename)
                dest_file += '.hdf5'

                utils.ensure_dir(map_gmm_dir)

                # load the GMM stats file
                #gmm_stats = bob.machine.GMMStats(bob.io.HDF5File(f_location))
                feature = bob.io.load(f_location)

                # extract i-vector
                logger.debug('gmm_stats type ' + str(type(feature)))
                #response = map_gmm_machine.forward(feature)
                response = map_gmm_machine(feature)

                # save them!
                bob.io.save(response, dest_file)

                logger.debug('saving map gmm response ' + dest_file)


            elif os.path.isdir(f_location):
                new_path = os.path.join(features_path, f)
                new_dest_dir = os.path.join(map_gmm_dir, f)
                map_gmm_machine_analysis(new_path, new_dest_dir, map_gmm_machine)

            else:
                logger.info('Unknown file type: ' + str(f_location))
                continue

def recursive_execute_gmms_on_machine(gmm_stats_path, dest_path, machine):
    for f in os.listdir(gmm_stats_path):
            gmm_stats_file = os.path.join(gmm_stats_path, f)
            logger.debug('extract_i_vectors: ' + gmm_stats_file)

            if utils.is_hdf5_file(gmm_stats_file):

                stripped_filename = utils.strip_filename(f)
                dest_file = os.path.join(dest_path, stripped_filename)
                dest_file += '.hdf5'

                utils.ensure_dir(dest_path)

                # load the GMM stats file
                gmm_stats = bob.machine.GMMStats(bob.io.HDF5File(gmm_stats_file))

                # extract i-vector
                output = machine.forward(gmm_stats)

                # save them!
                bob.io.save(output, dest_file)

                print 'savin ivec ' + dest_file


            elif os.path.isdir(gmm_stats_file):
                new_path = os.path.join(gmm_stats_path, f)
                new_dest_dir = os.path.join(dest_path, f)
                extract_i_vectors(new_path, new_dest_dir, machine)

            else:
                logger.info('Unknown file type: ' + str(gmm_stats_file))
                continue

def gen_map_gmm_trainer(ubm, relevance_factor, convergence_threshold=1e-5, max_iterations=200):
    trainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, False, False)
    trainer.convergence_threshold = convergence_threshold
    trainer.max_iterations = max_iterations
    trainer.set_prior_gmm(ubm)

    return trainer

def enroll(features, ubm, gmm_trainer):
    feature_set = numpy.vstack(features)
    # create a GMM from the UBM
    gmm = bob.machine.GMMMachine(ubm)

    # train the GMM
    gmm_trainer.train(gmm, feature_set)

    # return the resulting gmm
    return gmm

def gen_MAP_GMM_machines(ubm, all_features, gmm_trainer, dest_dir):

    for name, class_feat_files in all_features.iteritems():
        class_feats = load_mfcc_files(class_feat_files)
        class_gmm = enroll(class_feats, ubm, gmm_trainer)

        gmm_file = os.path.join(dest_dir, name + '.hdf5')
        class_gmm.save(bob.io.HDF5File(gmm_file, 'w'))





def gen_filedict(filelist):
    filedict = defaultdict(list)

    for f in filelist:
        name = utils.get_bird_name_from_file_name(f)
        filedict[name].append(f)

    return filedict



def gen_kmeans(training_set, number_of_gaussians):
    logger.info('Generating k-means.')
    #training_set = numpy.vstack(train_features)
    feature_dimensions = input_size = training_set.shape[1]

    # create the KMeans and UBM machine
    kmeans = bob.machine.KMeansMachine(number_of_gaussians, feature_dimensions)


    # create the KMeansTrainer
    kmeans_trainer = bob.trainer.KMeansTrainer()

    # train using the KMeansTrainer
    kmeans_trainer.train(kmeans, training_set)

    return kmeans

def gen_ubm(kmeans, training_set, trainer_convergence_threshold, trainer_max_iterations):
    logger.info('Generating UBM')
    #training_set = numpy.vstack(train_features)
    feature_dimensions = input_size = training_set.shape[1]

    [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(training_set)
    means = kmeans.means

    ubm = bob.machine.GMMMachine(kmeans.dim_c, kmeans.dim_d)

    # initialize the GMM
    ubm.means = means
    ubm.variances = variances
    ubm.weights = weights

    # train the GMM
    trainer = bob.trainer.ML_GMMTrainer()
    trainer.convergence_threshold = trainer_convergence_threshold
    trainer.max_iterations = trainer_max_iterations
    trainer.train(ubm, training_set)

    return ubm

#1. Extract and save features from training files
# recursively_extract_features(train_sounds_path, train_features_path)
#
# #2. Extract and save features from evaluation files
# recursively_extract_features(eval_sounds_path, eval_features_path)
#
# #path = '/home/timo/temp/dest/valid'
# #dest = '/Tavara/Ohjelmointi/BirDetect/birds/features'
#
#
# #3. Read training features to array
# train_features = recursive_load_mfcc_files(train_features_path)
#
#
# #4. array is converted to multidimensional ndarray
# train_features = numpy.vstack(train_features)
#
# #5. Clustering the train data using k-means and save result
# kmeans = train_k_means(train_features, num_gauss)
# kmeans.save(bob.io.HDF5File(kmeans_file, 'w'))
#
#
# #6. Create the universal background model
#
#
# #7. Train ubm-gmm and save result
# ubm = train_ubm_gmm_with_features(kmeans, train_features, ubm_convergence_threshold, ubm_max_iterations)
# #Save ubm
# ubm.save(bob.io.HDF5File(ubm_file, "w"))
#
# #8. Compute GMM sufficient statistics for both training and eval sets
# dim = train_features.shape[1]
# compute_gmm_sufficient_statistics(ubm, num_gauss, dim, train_features_path, gmm_stats_path)
#
# #9. Training TV and Sigma matrices
# gmm_train_stats = recursive_load_gmm_stats(gmm_stats_path)
#
# ivec_machine = train_tv(gmm_train_stats, ubm)
#
# #save the TV matrix
# ivec_machine.save(bob.io.HDF5File(tv_file, 'w'))
#
# #10. Extract i-vectors of the eval set..."
# extract_i_vectors(gmm_stats_path, ivec_dir, ivec_machine)
#
#
#
# #11. Write score file
#
# #Map adaptointi: https://www.idiap.ch/software/bob/docs/releases/last/sphinx/html/TutorialsTrainer.html
#
