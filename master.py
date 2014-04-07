# coding=utf-8
import numpy
import os
import bob
import logging
import split_wavs, analyze, utils

__author__ = 'Timo Mikkilä'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = os.getcwd()
print 'base path: ' + base_path

orig_sound_path = os.path.join(base_path, 'snd_orig')
sound_path = os.path.join(base_path, 'snd')

invalid_snd_path = os.path.join(base_path, 'snd_split_invalid')
split_plot_path = os.path.join(base_path, 'snd_split_plot')

train_sounds_path = os.path.join(sound_path, 'train')
eval_sounds_path = os.path.join(sound_path, 'eval')

analyze_path = os.path.join(base_path, 'analyze')

features_path = os.path.join(analyze_path, 'features')

train_features_path = os.path.join(features_path, 'train')
eval_features_path = os.path.join(features_path, 'eval')

mfcc_path = os.path.join(analyze_path, 'mfcc')
kmeans_file = os.path.join(analyze_path, 'kmeans.hdf5')
ubm_file = os.path.join(analyze_path, 'ubm.hdf5')
gmm_features_path = os.path.join(analyze_path, 'gmm_stats')

gmm_train_features_path = os.path.join(gmm_features_path, 'train')
gmm_eval_features_path = os.path.join(gmm_features_path, 'eval')

ivec_machine_file = os.path.join(analyze_path, 'tv.hdf5')
ivec_dir = os.path.join(analyze_path, 'ivectors')

gmm_machine_file = os.path.join(analyze_path, 'gmm_machine.hdf5')


eval_score_file = os.path.join(analyze_path, 'score-eval.txt')

map_gmm_relevance = 4
map_gmm_convergence_threshold = 1e-5
map_gmm_max_iterations = 200


num_gauss = 16# 128 512

ubm_convergence_threshold = 1e-4
ubm_max_iterations = 10




#EXECUTE!!

#Split audio files

if not os.path.exists(sound_path):
    split_wavs.recursively_split_wav_files(orig_sound_path, sound_path, invalid_snd_path)

#Make plots of splits for later analysis
#TODO: integrate to splitting
#TODO:Set proper font size for plotting

if not os.path.exists(split_plot_path):
    split_wavs.recursively_plot_wav_files(orig_sound_path, split_plot_path)

##RUN ANALYSIS##

#1. Extract and save features from training files

if not os.path.exists(train_features_path):
    analyze.recursively_extract_features(train_sounds_path, train_features_path)

#2. Extract and save features from evaluation files
if not os.path.exists(eval_features_path):
    analyze.recursively_extract_features(eval_sounds_path, eval_features_path)

#3. Read training features to array
logger.info('Load train features')
#analyze.recursive_test_wavs_for_nan(sound_path)
#analyze.recursive_test_for_nan(train_features_path)

train_features = analyze.recursive_load_mfcc_files(train_features_path)

#4. array is converted to multidimensional ndarray
logger.info('Convert train features array')
train_features = numpy.vstack(train_features)

#5. Clustering the train data using k-means and save result
logger.info('Train k-means')

kmeans = None

if os.path.isfile(kmeans_file):
    logger.info('K-Means file exists (' + kmeans_file + '). No new K-Means is generated.')
    kmeans_hdf5 = bob.io.HDF5File(kmeans_file)
    kmeans = bob.machine.KMeansMachine(kmeans_hdf5)
    logger.info('K-Means file loaded')
else:
    kmeans = analyze.train_k_means(train_features, num_gauss)
    logger.info('save K-Means to file: ' + kmeans_file)
    kmeans.save(bob.io.HDF5File(kmeans_file, 'w'))


#6. Create the universal background model


#7. Train ubm-gmm and save result
logger.info('Train UBM-GMM')

ubm = None

#If UBM file exists, open it
if os.path.isfile(ubm_file):
    logger.info('UBM file exists (' + ubm_file + '). No new UBM is generated.')
    ubm_hdf5 = bob.io.HDF5File(ubm_file)
    ubm = bob.machine.GMMMachine(ubm_hdf5)
    logger.info('UBM file loaded')
else:
    logger.info('No UBM file found (' + ubm_file + '). New UBM is generated.')
    ubm = analyze.train_ubm_gmm_with_features(kmeans, train_features, num_gauss, ubm_convergence_threshold, ubm_max_iterations)
#Save ubm
    logger.info('Save UBM to file: ' + ubm_file)
    ubm.save(bob.io.HDF5File(ubm_file, "w"))


#8. Compute GMM sufficient statistics for both training and eval sets
dim = train_features.shape[1]

if os.path.exists(gmm_train_features_path):
    logger.info('GMM train stats path exists ('+ gmm_train_features_path +') Skipping... ')
else:
    logger.info('Calculating GMM train stats')
    analyze.compute_gmm_sufficient_statistics(ubm, num_gauss, dim, train_features_path, gmm_train_features_path)

if os.path.exists(gmm_eval_features_path):
    logger.info('GMM evaluation stats path exists ('+ gmm_eval_features_path +') Skipping... ')
else:
    logger.info('Calculating GMM evaluation stats')
    analyze.compute_gmm_sufficient_statistics(ubm, num_gauss, dim, train_features_path, gmm_eval_features_path)

#9. Training TV and Sigma matrices
gmm_train_stats = analyze.recursive_load_gmm_stats(gmm_train_features_path)

ivec_machine = None

if os.path.isfile(ivec_machine_file):
    logger.info('TV matrix file exists (' + ivec_machine_file + '). No new TV matrix is generated.')
    tv_hdf5 = bob.io.HDF5File(ivec_machine_file)
    ivec_machine = bob.machine.IVectorMachine(tv_hdf5)
    logger.info('TV matrix file loaded')
else:
    logger.info('Generating TV matrix.')
    ivec_machine = analyze.train_tv(gmm_train_stats, ubm)
    logger.info('TV matrix generated.')
    #save the TV matrix
    logger.info('Saving TV matrix to ' + ivec_machine_file)
    ivec_machine.save(bob.io.HDF5File(ivec_machine_file, 'w'))

#10. Extract i-vectors of the eval set..."

logger.info('10.  Extract i-vectors of the evaluation set.')
if os.path.exists(ivec_dir):
    logger.info('Ivectors  for evaluation set found so no ivectors are calculated.')
else:
    logger.info('Calculating ivectors for evaluation set')
    analyze.extract_i_vectors(gmm_eval_features_path, ivec_dir, ivec_machine)


#Load all ivec file names to array
#Compare files to each other and write to score file
files = analyze.recursive_find_all_files(ivec_dir, '.hdf5')
f = open(eval_score_file, 'w')

for f1 in files:
    for f2 in files:
        f1_loaded = bob.io.load(f1)
        f2_loaded = bob.io.load(f2)
        score = utils.cosine_score(f1_loaded, f2_loaded)
        f.write('\"'+f1[len(ivec_dir)+1:] + '\",\"' + f2[len(ivec_dir)+1:] + '\",\"' + str(score) + '\"\n')

f.close()


# gmm_machine = None
#
# if os.path.isfile(gmm_machine_file):
#     logger.info('GMM Machine file exists (' + gmm_machine_file + '). Loading MAP GMM Machine from file instead of generating it.')
#     gmm_machine_hdf5 = bob.io.HDF5File(gmm_machine_file)
#     gmm_machine = bob.machine.bob.machine.GMMMachine(gmm_machine_hdf5)
#     logger.info('MAP GMM Machine file loaded')
# else:
#     logger.info('Generating MAP GMM Machine.')
#     gmm_machine = analyze.gen_MAP_GMM_machine(ubm, train_features, dim, map_gmm_relevance, map_gmm_convergence_threshold, map_gmm_max_iterations)
#     logger.info('TV matrix generated.')
#     #save the TV matrix
#     logger.info('Saving MAP GMM Machine to ' + gmm_machine_file)
#     gmm_machine.save(bob.io.HDF5File(gmm_machine_file, 'w'))

#Generate map adaptation trainer
#gmm_machine = analyze.gen_MAP_GMM_machine(ubm, train_features, dim, map_gmm_relevance, map_gmm_convergence_threshold, map_gmm_max_iterations)
#save map adaptation trainer
#gmm_machine.save(bob.io.HDF5File(gmm_machine_file, 'w'))