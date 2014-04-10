# coding=utf-8
import numpy
import os
import bob
import logging
import time
import analyze, utils, evaluate
import logging.config
import argparse

__author__ = 'Timo Mikkilä'

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_gauss', help='Number of gaussian used in calculations', default=16, type=int)
parser.add_argument('--snd_path', help='Location of wav files. Must contain train and eval folders', default='snd')
parser.add_argument('--dest_path', help='Location where all data is written to', default='dest')

args = parser.parse_args()

dest_path = os.path.abspath(args.dest_path)

log_file_locator = lambda *x: os.path.join(dest_path, *x)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },

        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": log_file_locator('info.log'),
            "maxBytes": "10485760",
            "backupCount": "20",
            "encoding": "utf8"
        },

        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "simple",
            "filename": log_file_locator('errors.log'),
            "maxBytes": "10485760",
            "backupCount": "20",
            "encoding": "utf8"
        },

        "debug_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": log_file_locator('debug.log'),
            "maxBytes": "10485760",
            "backupCount": "20",
            "encoding": "utf8"
        }
    },

    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": "no"
        }
    },

    "root": {
        "level": "DEBUG",
        "handlers": ["console", "info_file_handler", "error_file_handler", "debug_file_handler"]
    }
}



def setup_logging(dest_path, logging_json, default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration

    """

    if logging_json is not None:
        #config = json.loads(logging_json)
        logging.config.dictConfig(logging_json)
    else:
        logging.basicConfig(level=default_level)





#setup_logging(dest_path)
#logger = logging.getLogger(__name__)
#

info_log_file = os.path.join(dest_path, 'info.log')
debug_log_file = os.path.join(dest_path, 'debug.log')
# create a file handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# info_handler = logging.FileHandler(info_log_file)
# info_handler.setLevel(logging.INFO)
# debug_handler = logging.FileHandler(debug_log_file)
# debug_handler.setLevel(logging.DEBUG)
# # create a logging format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# info_handler.setFormatter(formatter)
# debug_handler.setFormatter(formatter)
# # add the handlers to the logger
# logger.addHandler(console_handler)
# logger.addHandler(info_handler)
# logger.addHandler(debug_handler)

log_conf_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logging.json')
setup_logging(dest_path, LOGGING)


sound_path = os.path.abspath(args.snd_path)
logger.info('sound_path=' + sound_path)

train_sounds_path = os.path.join(sound_path, 'train')
logger.info('train_sounds_path=' + train_sounds_path)
eval_sounds_path = os.path.join(sound_path, 'eval')
logger.info('eval_sounds_path=' + eval_sounds_path)


logger.info('dest_path=' + dest_path)

features_path = os.path.join(dest_path, 'features')
logger.info('features_path=' + features_path)

train_features_path = os.path.join(features_path, 'train')
logger.info('train_features_path=' + train_features_path)

eval_features_path = os.path.join(features_path, 'eval')
logger.info('eval_features_path=' + eval_features_path)

mfcc_path = os.path.join(dest_path, 'mfcc')
logger.info('mfcc_path=' + mfcc_path)
kmeans_file = os.path.join(dest_path, 'kmeans.hdf5')
logger.info('kmeans_file=' + kmeans_file)
ubm_file = os.path.join(dest_path, 'ubm.hdf5')
logger.info('ubm_file=' + ubm_file)
gmm_features_path = os.path.join(dest_path, 'gmm_stats')
logger.info('gmm_features_path=' + gmm_features_path)

gmm_stats_train_path = os.path.join(gmm_features_path, 'train')
logger.info('gmm_train_features_path=' + gmm_stats_train_path)
gmm_stats_eval_path = os.path.join(gmm_features_path, 'eval')
logger.info('gmm_eval_features_path=' + gmm_stats_eval_path)

ivec_machine_file = os.path.join(dest_path, 'tv.hdf5')
logger.info('ivec_machine_file=' + ivec_machine_file)
ivec_dir = os.path.join(dest_path, 'ivectors')
logger.info('ivec_dir=' + ivec_dir)

gmm_machine_file = os.path.join(dest_path, 'gmm_machine.hdf5')
logger.info('gmm_machine_file=' + gmm_machine_file)
map_gmm_dir = os.path.join(dest_path, 'map_gmm')
logger.info('map_gmm_dir=' + map_gmm_dir)

eval_ivector_score_file = os.path.join(dest_path, 'score-ivector-eval.txt')
logger.info('eval_ivector_score_file=' + eval_ivector_score_file)
eval_map_gmm_score_file = os.path.join(dest_path, 'score-map_gmm-eval.txt')
logger.info('eval_map_gmm_score_file=' + eval_map_gmm_score_file)

test_roc_eval_map_gmm_file = os.path.join(dest_path, 'gmm_adapted_roc.png')
logger.info('test_roc_eval_map_gmm_file=' + test_roc_eval_map_gmm_file)
test_det_eval_map_gmm_file = os.path.join(dest_path, 'gmm_adapted_det.png')
logger.info('test_det_eval_map_gmm_file=' + test_det_eval_map_gmm_file)

test_roc_eval_ivec_file = os.path.join(dest_path, 'ivec_roc.png')
logger.info('test_roc_eval_ivec_file=' + test_roc_eval_ivec_file)
test_det_eval_ivec_file = os.path.join(dest_path, 'ivec_det.png')
logger.info('test_det_eval_ivec_file=' + test_det_eval_ivec_file)

logger.info('Other parameters:')
num_gauss = args.num_gauss #defaults to 16 #19# 128 512
logger.info('num_gauss=' + str(num_gauss))

#MFCC feature extraction parameters
mfcc_wl=20 #The window length in milliseconds
mfcc_ws=10 # The window shift of the in milliseconds
mfcc_nf=24 # The number of filter bands
mfcc_nceps=19 # The number of cepstral coefficients
mfcc_fmin=0. # The minimal frequency of the filter bank
mfcc_fmax=4000. # The maximal frequency of the filter bank
mfcc_d_w=2 # The delta value used to compute 1st and 2nd derivatives
mfcc_pre=0.97 # The coefficient used for the pre-emphasis
mfcc_mel=True # Tell whether MFCC or LFCC are extracted

logger.info('MFCC feature extraction parameters:')
logger.info('mfcc_wl=' + str(mfcc_wl))
logger.info('mfcc_ws=' + str(mfcc_ws))
logger.info('mfcc_nf=' + str(mfcc_nf))
logger.info('mfcc_nceps=' + str(mfcc_nceps))
logger.info('mfcc_fmin=' + str(mfcc_fmin))
logger.info('mfcc_fmax=' + str(mfcc_fmax))
logger.info('mfcc_d_w=' + str(mfcc_d_w))
logger.info('mfcc_pre=' + str(mfcc_pre))
logger.info('mfcc_mel=' + str(mfcc_mel))

#Kmeans machine parameters
kmeans_num_gauss= num_gauss
kmeans_dim= 19
logger.info('Kmeans machine parameters:')
logger.info('kmeans_num_gauss=' + str(kmeans_num_gauss))
logger.info('kmeans_dim=' + str(kmeans_dim))

#GMM stats parameters
gmm_stat_num_gauss = num_gauss
gmm_stat_dim = 19
logger.info('GMM stats parameters:')
logger.info('gmm_stat_num_gauss=' + str(gmm_stat_num_gauss))
logger.info('gmm_stat_dim=' + str(gmm_stat_dim))

#Ivector machine parameters
ivec_machine_variance_treshold=1e-5
ivec_machine_dim = 19
logger.info('Ivector machine parameters:')
logger.info('ivec_machine_dim=' + str(ivec_machine_dim))
logger.info('ivec_machine_variance_treshold=' + str(ivec_machine_variance_treshold))

#Ivector trainer parameters
ivec_trainer_max_iterations=10
ivec_trainer_update_sigma=True
logger.info('Ivector trainer parameters:')
logger.info('ivec_trainer_max_iterations=' + str(ivec_trainer_max_iterations))
logger.info('ivec_trainer_update_sigma=' + str(ivec_trainer_update_sigma))

#Parameters for map gmm trainer
map_gmm_relevance_factor = 4
map_gmm_convergence_threshold = 1e-5
map_gmm_max_iterations = 200
logger.info('Map gmm trainer parameters:')
logger.info('map_gmm_relevance_factor=' + str(map_gmm_relevance_factor))
logger.info('map_gmm_convergence_threshold=' + str(map_gmm_convergence_threshold))
logger.info('map_gmm_max_iterations=' + str(map_gmm_max_iterations))

#Parameters for map adapted gmm machine
gmm_adapted_num_gauss = num_gauss
gmm_adapted_dim = 19
logger.info('Map adapted gmm machine parameters:')
logger.info('gmm_adapted_num_gauss=' + str(gmm_adapted_num_gauss))
logger.info('gmm_adapted_dim=' + str(gmm_adapted_dim))

ubm_gmm_num_gauss = num_gauss
ubm_gmm_dim = 19
ubm_convergence_threshold = 1e-4
ubm_max_iterations = 10
logger.info('UBM parameters:')
logger.info('ubm_gmm_num_gauss=' + str(ubm_gmm_num_gauss))
logger.info('ubm_gmm_dim=' + str(ubm_gmm_dim))
logger.info('ubm_convergence_threshold=' + str(ubm_convergence_threshold))
logger.info('ubm_max_iterations=' + str(ubm_max_iterations))




def gen_score_file(files, score_file):
    #Load all ivec file names to array
    #Compare files to each other and write to score file
    logger.debug('Generating score file to ' + score_file)
    f = open(score_file, 'w')

    for f1 in files:
        for f2 in files:
            f1_loaded = bob.io.load(f1)
            f2_loaded = bob.io.load(f2)
            f1_loaded = numpy.linalg.norm(f1_loaded)
            f2_loaded = numpy.linalg.norm(f2_loaded)
            score = numpy.dot(f1_loaded, f2_loaded)

            #score = utils.cosine_score(f1_loaded, f2_loaded)
            f.write('\"'+f1[len(ivec_dir)+1:] + '\",\"' + f2[len(ivec_dir)+1:] + '\",\"' + str(score) + '\"\n')

    f.close()

def evaluate_score_file(score_file, roc_file, det_file):
    logger.info('Evaluating ' + score_file)

    logger.info('Finding negatives and positives from score file')
    negatives, positives = evaluate.parse_scores_from_file(score_file)

    logger.info('Found ' + str(len(negatives)) + ' negatives and ' + str(len(positives)) + ' positives')

    eer_rocch = bob.measure.eer_rocch(negatives, positives)
    logger.info('eer_rocch=' + str(eer_rocch))
    eer_threshold = bob.measure.eer_threshold(negatives, positives)
    logger.info('eer_threshold=' + str(eer_threshold))

    logger.info('Generating ROC curve to ' + roc_file)
    evaluate.gen_roc_curve(negatives, positives, roc_file)

    logger.info('Generating DET curve to ' + det_file)
    evaluate.gen_det_curve(negatives, positives, det_file)

total_start_time = time.clock()
#EXECUTE!!

#Split audio files
# if args.split:
#     if os.path.exists(sound_path):
#         logger.info('No splitting is done as split destination directory ('+ sound_path +') already exists')
#     else:
#         logger.info('Splitting wavs from ' + orig_sound_path +' to ' + sound_path)
#         split_wavs.recursively_split_wav_files(orig_sound_path, sound_path, invalid_snd_path)
#         split_wavs.recursively_plot_wav_files(orig_sound_path, split_plot_path)


##RUN ANALYSIS##
logger.info('STARTING PROCESSING')
logger.info('Extracting features')

#1. Extract and save features from training files
if os.path.exists(train_features_path):
    logger.warn('Features are not extracted for train data as folder already exists('+train_features_path+').')
else:
    logger.info('Extracting mfcc features from train data')
    logger.debug('From ' + train_sounds_path + ' to ' + train_features_path)
    analyze.recursively_extract_features(train_sounds_path, train_features_path, mfcc_wl, mfcc_ws, mfcc_nf, mfcc_nceps, mfcc_fmin, mfcc_fmax, mfcc_d_w, mfcc_pre, mfcc_mel)

#2. Extract and save features from evaluation files
if os.path.exists(eval_features_path):
    logger.warn('Features are not extracted for evaluation data as folder already exists ('+eval_features_path+').')
else:
    logger.info('Extracting mfcc features from evaluation data')
    logger.debug('From ' + eval_sounds_path + ' to ' + eval_features_path)
    analyze.recursively_extract_features(eval_sounds_path, eval_features_path, mfcc_wl, mfcc_ws, mfcc_nf, mfcc_nceps, mfcc_fmin, mfcc_fmax, mfcc_d_w, mfcc_pre, mfcc_mel)

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
    kmeans = analyze.train_kmeans_machine(train_features, kmeans_num_gauss, kmeans_dim)
    logger.info('save K-Means to file: ' + kmeans_file)
    kmeans.save(bob.io.HDF5File(kmeans_file, 'w'))

#analyze.test_array_for_nan(kmeans.means)
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
    ubm = analyze.train_ubm_gmm_with_features(kmeans, train_features, ubm_gmm_num_gauss, ubm_gmm_dim, ubm_convergence_threshold, ubm_max_iterations)
#Save ubm
    logger.info('Save UBM to file: ' + ubm_file)
    ubm.save(bob.io.HDF5File(ubm_file, "w"))


logger.info('Compute GMM sufficient statistics for both training and eval sets')
#8. Compute GMM sufficient statistics for both training and eval sets
dim = train_features.shape[1]
logger.info('Train features dimension = ' + str(dim))

if os.path.exists(gmm_stats_train_path):
    logger.info('GMM train stats path exists ('+ gmm_stats_train_path +') Skipping... ')
else:
    logger.info('Calculating GMM train stats')
    analyze.compute_gmm_sufficient_statistics(ubm, gmm_stat_num_gauss, gmm_stat_dim, train_features_path, gmm_stats_train_path)

if os.path.exists(gmm_stats_eval_path):
    logger.info('GMM evaluation stats path exists ('+ gmm_stats_eval_path +') Skipping... ')
else:
    logger.info('Calculating GMM evaluation stats')
    analyze.compute_gmm_sufficient_statistics(ubm, num_gauss, dim, train_features_path, gmm_stats_eval_path)

#9. Training TV and Sigma matrices
logger.info('Training ivector machine')
gmm_train_stats = analyze.recursive_load_gmm_stats(gmm_stats_train_path)

ivec_machine = None

if os.path.isfile(ivec_machine_file):
    logger.info('Ivector machine file exists (' + ivec_machine_file + '). No new ivector machine is generated.')
    tv_hdf5 = bob.io.HDF5File(ivec_machine_file)
    ivec_machine = bob.machine.IVectorMachine(tv_hdf5)
    logger.info('Ivector machine loaded')
else:
    logger.info('Generating ivector machine.')
    ivec_machine = analyze.gen_ivec_machine(gmm_train_stats, ubm, ivec_machine_dim, ivec_machine_variance_treshold, ivec_trainer_max_iterations, ivec_trainer_update_sigma)
    logger.info('Ivector machine generated.')
    #save the TV matrix
    logger.info('Saving ivector machine to ' + ivec_machine_file)
    ivec_machine.save(bob.io.HDF5File(ivec_machine_file, 'w'))

#10. Extract i-vectors of the eval set..."

logger.info('Extract i-vectors of the evaluation set.')
if os.path.exists(ivec_dir):
    logger.info('Ivectors  for evaluation set found so no ivectors are calculated.')
else:
    logger.info('Calculating ivectors for evaluation set')
    analyze.extract_i_vectors(gmm_stats_eval_path, ivec_dir, ivec_machine)

logger.info('Generate score file for ivector comparisons')
if os.path.exists(eval_ivector_score_file):
    logger.info('Score file for ivector analysis already exists in '+eval_ivector_score_file+'. No new score file is generated.')
else:
    logger.info('Generating score file for ivector analysis')
    files = analyze.recursive_find_all_files(ivec_dir, '.hdf5')
    gen_score_file(files, eval_ivector_score_file)
    logger.info('Score file generated to: ' + eval_ivector_score_file)


logger.info('Train GMM Machine with Map adaptation')
gmm_machine = None

if os.path.isfile(gmm_machine_file):
    logger.info('GMM Machine file exists (' + gmm_machine_file + '). Loading MAP GMM Machine from file instead of generating it.')
    gmm_machine_hdf5 = bob.io.HDF5File(gmm_machine_file)
    gmm_machine = bob.machine.GMMMachine(gmm_machine_hdf5)
    logger.info('MAP GMM Machine file loaded')
else:
    logger.info('Generating MAP GMM Machine.')
    gmm_machine = analyze.gen_MAP_GMM_machine(ubm, train_features, gmm_adapted_num_gauss, gmm_adapted_dim, map_gmm_relevance_factor, map_gmm_convergence_threshold, map_gmm_max_iterations)
    logger.info('TV matrix generated.')
    #save the TV matrix
    logger.info('Saving MAP GMM Machine to ' + gmm_machine_file)
    gmm_machine.save(bob.io.HDF5File(gmm_machine_file, 'w'))

# logger.info('Run evaluation set through map gmm machine.')
# if os.path.exists(map_gmm_dir):
#     logger.info('map gmm path already exists.')
# else:
#     logger.info('Calculating map gmm for evaluation set')
#     analyze.map_gmm_machine_analysis(eval_features_path, map_gmm_dir, gmm_machine)
#
# logger.info('Generate score file for gmm comparisons')
#
# if os.path.exists(eval_map_gmm_score_file):
#     logger.info('Score file for map gmm analysis already exists in '+eval_map_gmm_score_file+'. No new score file is generated.')
# else:
#     logger.info('Generating score file for map_gmm analysis')
#     files = analyze.recursive_find_all_files(ivec_dir, '.hdf5')
#     gen_score_file(files, eval_map_gmm_score_file)
#     logger.info('Score file generated to: ' + eval_map_gmm_score_file)


#EVALUATE RESULTS
logger.info('Evaluating results')
logger.info('Evaluating ivector results')
evaluate_score_file(eval_ivector_score_file,test_roc_eval_ivec_file, test_det_eval_ivec_file)
# logger.info('Evaluating map adapted gmm results')
# evaluate_score_file(eval_map_gmm_score_file, test_roc_eval_map_gmm_file, test_det_eval_map_gmm_file)

total_end_time = time.clock()

logger.info('Total time elapsed: ' + str(total_end_time - total_start_time))