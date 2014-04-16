# coding=utf-8
import numpy
import os
import bob
import logging
import time
from bunch import Bunch
import sys
import analyze, utils, evaluate
import ivector
import ubmgmm
import logging.config
import argparse
import yaml
import traceback
#import utils.Bunch as Bunch
from matplotlib import pyplot

__author__ = 'Timo Mikkilä'

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

parameters = {}

def load_parameters(param_file):
    if not os.path.isfile(param_file):
        logger.error('Error no such parameter file:' + param_file)
        exit(128)
    f=open(param_file)
    global parameters
    new_params = yaml.load(f)
    parameters = dict(parameters.items() + new_params.items())

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('run_mode', help='What to do', choices=['ivector', 'ubm-gmm'])
parser.add_argument('--num_gauss', help='Number of gaussian used in calculations', default=16, type=int)
parser.add_argument('--snd_path', help='Location of wav files. Must contain train and eval folders', default='snd')
parser.add_argument('--dest_path', help='Location where all data is written to', default='dest')
parser.add_argument('--params_file', help='Parameter file')


args = parser.parse_args()

paths = Bunch()

paths.dest = os.path.abspath(args.dest_path)

paths.sound = os.path.abspath(args.snd_path)
logger.info('sound=' + paths.sound)

paths.train_sounds = os.path.join(paths.sound, 'train')
logger.info('train_sounds=' + paths.train_sounds)
paths.eval_sounds = os.path.join(paths.sound, 'eval')
logger.info('eval_sounds=' + paths.eval_sounds)


logger.info('dest=' + paths.dest)

paths.eval_log_file = os.path.join(paths.dest, 'eval.txt')

paths.features = os.path.join(paths.dest, 'features')
logger.info('features=' + paths.features)

paths.train_features = os.path.join(paths.features, 'train')
logger.info('train_features=' + paths.train_features)

paths.eval_features = os.path.join(paths.features, 'eval')
logger.info('eval_features=' + paths.eval_features)

paths.mfcc = os.path.join(paths.dest, 'mfcc')
logger.info('mfcc=' + paths.mfcc)
paths.kmeans_file = os.path.join(paths.dest, 'kmeans.hdf5')
logger.info('kmeans_file=' + paths.kmeans_file)
paths.ubm_file = os.path.join(paths.dest, 'ubm.hdf5')
logger.info('ubm_file=' + paths.ubm_file)
paths.gmm_features = os.path.join(paths.dest, 'gmm_stats')
logger.info('gmm_features=' + paths.gmm_features)

paths.gmm_stats_train = os.path.join(paths.gmm_features, 'train')
logger.info('gmm_train_features=' + paths.gmm_stats_train)
paths.gmm_stats_eval = os.path.join(paths.gmm_features, 'eval')
logger.info('gmm_eval_features=' + paths.gmm_stats_eval)

paths.ivec_machine_file = os.path.join(paths.dest, 'tv.hdf5')
logger.info('ivec_machine_file=' + paths.ivec_machine_file)
paths.ivec_dir = os.path.join(paths.dest, 'ivectors')
logger.info('ivec_dir=' + paths.ivec_dir)

paths.gmm_machine_file = os.path.join(paths.dest, 'gmm_machine.hdf5')
logger.info('gmm_machine_file=' + paths.gmm_machine_file)
paths.map_gmm_dir = os.path.join(paths.dest, 'map_gmm')
logger.info('map_gmm_dir=' + paths.map_gmm_dir)

paths.eval_ivector_score_file = os.path.join(paths.dest, 'score-ivector-eval.txt')
logger.info('eval_ivector_score_file=' + paths.eval_ivector_score_file)
paths.eval_map_gmm_score_file = os.path.join(paths.dest, 'score-map_gmm-eval.txt')
logger.info('eval_map_gmm_score_file=' + paths.eval_map_gmm_score_file)

paths.test_roc_eval_map_gmm_file = os.path.join(paths.dest, 'gmm_adapted_roc.png')
logger.info('test_roc_eval_map_gmm_file=' + paths.test_roc_eval_map_gmm_file)
paths.test_det_eval_map_gmm_file = os.path.join(paths.dest, 'gmm_adapted_det.png')
logger.info('test_det_eval_map_gmm_file=' + paths.test_det_eval_map_gmm_file)

paths.test_roc_eval_ivec_file = os.path.join(paths.dest, 'ivec_roc.png')
logger.info('test_roc_eval_ivec_file=' + paths.test_roc_eval_ivec_file)
paths.test_det_eval_ivec_file = os.path.join(paths.dest, 'ivec_det.png')
logger.info('test_det_eval_ivec_file=' + paths.test_det_eval_ivec_file)

paths.conf_default_file = os.path.join(os.path.dirname(__file__), 'default.cfg')

log_file_locator = lambda *x: os.path.join(paths.dest, *x)

#Load default parameters
load_parameters(paths.conf_default_file)

if args.params_file is not None:
    load_parameters(args.params_file)

params = Bunch(parameters)

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



info_log_file = os.path.join(paths.dest, 'info.log')
debug_log_file = os.path.join(paths.dest, 'debug.log')
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
setup_logging(paths.dest, LOGGING)

logger.info('Loading parameters from ' + paths.conf_default_file)
logger.info('Parameters loaded: ' + str(parameters))

logger.info('Other parameters:')
#num_gauss = args.num_gauss #defaults to 16 #19# 128 512
#logger.info('num_gauss=' + str(num_gauss))
if(args.num_gauss is not None):
    logger.warning('Overriding number of gaussians (' + str(params.number_of_gaussians) + ' -> ' + str(args.num_gauss) + ')')
    params.number_of_gaussians = args.num_gauss
#parameters['kmeans_num_gauss'] = num_gauss
#parameters['gmm_stat_num_gauss'] = num_gauss
#parameters['gmm_adapted_num_gauss'] = num_gauss


#MFCC feature extraction parameters
# parameters['mfcc_wl']=20 #The window length in milliseconds
# parameters['mfcc_ws']=10 # The window shift of the in milliseconds
# parameters['mfcc_nf']=24 # The number of filter bands
# parameters['mfcc_nceps']=19 # The number of cepstral coefficients
# parameters['mfcc_fmin']=0. # The minimal frequency of the filter bank
# parameters['mfcc_fmax']=4000. # The maximal frequency of the filter bank
# parameters['mfcc_d_w']=2 # The delta value used to compute 1st and 2nd derivatives
# parameters['mfcc_pre']=0.97 # The coefficient used for the pre-emphasis
# parameters['mfcc_mel']=True # Tell whether MFCC or LFCC are extracted
#
# logger.info('MFCC feature extraction parameters:')
# logger.info('parameters[\'mfcc_wl\']=' + str(parameters['mfcc_wl']))
# logger.info('parameters[\'mfcc_ws\']=' + str(parameters['mfcc_ws']))
# logger.info('parameters[\'mfcc_nf\']=' + str(parameters['mfcc_nf']))
# logger.info('parameters[\'mfcc_nceps\']=' + str(parameters['mfcc_nceps']))
# logger.info('parameters[\'mfcc_fmin\']=' + str(parameters['mfcc_fmin']))
# logger.info('parameters[\'mfcc_fmax\']=' + str(parameters['mfcc_fmax']))
# logger.info('parameters[\'mfcc_d_w\']=' + str(parameters['mfcc_d_w']))
# logger.info('parameters[\'mfcc_pre\']=' + str(parameters['mfcc_pre']))
# logger.info('parameters[\'mfcc_mel\']=' + str(parameters['mfcc_mel']))
#
# #Kmeans machine parameters
# # parameters['kmeans_num_gauss']= num_gauss
# # parameters['kmeans_dim']= 19
# logger.info('Kmeans machine parameters:')
# logger.info('parameters['kmeans_num_gauss']=' + str(parameters['kmeans_num_gauss']))
# logger.info('parameters['kmeans_dim']=' + str(parameters['kmeans_dim']))
#
# #GMM stats parameters
# # parameters['gmm_stat_num_gauss'] = num_gauss
# # parameters['gmm_stat_dim'] = 19
# logger.info('GMM stats parameters:')
# logger.info('parameters['gmm_stat_num_gauss']=' + str(parameters['gmm_stat_num_gauss']))
# logger.info('parameters['gmm_stat_dim']=' + str(parameters['gmm_stat_dim']))
#
# #Ivector machine parameters
# # parameters['ivec_machine_variance_treshold']=1e-5
# # parameters['ivec_machine_dim'] = 19
# logger.info('Ivector machine parameters:')
# logger.info('parameters['ivec_machine_dim']=' + str(parameters['ivec_machine_dim']))
# logger.info('parameters['ivec_machine_variance_treshold']=' + str(parameters['parameters'][ivec_machine_variance_treshold))
#
# # #Ivector trainer parameters
# # parameters['ivec_trainer_max_iterations']=10
# # parameters['ivec_trainer_update_sigma']=True
# logger.info('Ivector trainer parameters:')
# logger.info('parameters['ivec_trainer_max_iterations']=' + str(parameters['ivec_trainer_max_iterations']))
# logger.info('parameters['ivec_trainer_update_sigma']=' + str(parameters['ivec_trainer_update_sigma']))
#
# # #Parameters for map gmm trainer
# # parameters['map_gmm_relevance_factor'] = 4
# # parameters['map_gmm_convergence_threshold'] = 1e-5
# # parameters['map_gmm_max_iterations'] = 200
# logger.info('Map gmm trainer parameters:')
# logger.info('parameters['map_gmm_relevance_factor']=' + str(parameters['map_gmm_relevance_factor']))
# logger.info('parameters['map_gmm_convergence_threshold']=' + str(parameters['map_gmm_convergence_threshold']))
# logger.info('parameters['map_gmm_max_iterations']=' + str(parameters['map_gmm_max_iterations']))
#
# # #Parameters for map adapted gmm machine
# # parameters['gmm_adapted_num_gauss'] = num_gauss
# # parameters['gmm_adapted_dim'] = 19
# logger.info('Map adapted gmm machine parameters:')
# logger.info('parameters['gmm_adapted_num_gauss']=' + str(parameters['gmm_adapted_num_gauss']))
# logger.info('parameters['gmm_adapted_dim']=' + str(parameters['gmm_adapted_dim']))
#
# # parameters['ubm_gmm_num_gauss'] = num_gauss
# # parameters['ubm_gmm_dim'] = 19
# # parameters['ubm_convergence_threshold'] = 1e-4
# # parameters['ubm_max_iterations'] = 10
# logger.info('UBM parameters:')
# logger.info('parameters['ubm_gmm_num_gauss']=' + str(parameters['ubm_gmm_num_gauss']))
# logger.info('parameters['ubm_gmm_dim']=' + str(parameters['ubm_gmm_dim']))
# logger.info('parameters['ubm_convergence_threshold']=' + str(parameters['ubm_convergence_threshold']))
# logger.info('parameters['ubm_max_iterations']=' + str(parameters['ubm_max_iterations']))


if args.run_mode == 'ubm-gmm':
    worker = ubmgmm.ubm_gmm_worker(paths, params)
elif args.run_mode == 'ivector':
    worker = ivector.ivector_worker(paths, params)
try:
    worker.run()
except:
    logging.exception('Something went wrong')

# total_start_time = time.clock()
# #EXECUTE!!
#
# #Split audio files
# # if args.split:
# #     if os.path.exists(sound_path):
# #         logger.info('No splitting is done as split destination directory ('+ sound_path +') already exists')
# #     else:
# #         logger.info('Splitting wavs from ' + orig_sound_path +' to ' + sound_path)
# #         split_wavs.recursively_split_wav_files(orig_sound_path, sound_path, invalid_snd_path)
# #         split_wavs.recursively_plot_wav_files(orig_sound_path, split_plot_path)
#
#
# ##RUN ANALYSIS##
# logger.info('STARTING PROCESSING')
# logger.info('Extracting features')
#
# #1. Extract and save features from training files
# if os.path.exists(train_features_path):
#     logger.warn('Features are not extracted for train data as folder already exists('+train_features_path+').')
# else:
#     logger.info('Extracting mfcc features from train data')
#     logger.debug('From ' + train_sounds_path + ' to ' + train_features_path)
#     analyze.recursively_extract_features(train_sounds_path, train_features_path, params.mfcc_wl, parameters['mfcc_ws'], parameters['mfcc_nf'], parameters['mfcc_nceps'], parameters['mfcc_fmin'], parameters['mfcc_fmax'], parameters['mfcc_d_w'], parameters['mfcc_pre'], parameters['mfcc_mel'])
#
# #2. Extract and save features from evaluation files
# if os.path.exists(eval_features_path):
#     logger.warn('Features are not extracted for evaluation data as folder already exists ('+eval_features_path+').')
# else:
#     logger.info('Extracting mfcc features from evaluation data')
#     logger.debug('From ' + eval_sounds_path + ' to ' + eval_features_path)
#     analyze.recursively_extract_features(eval_sounds_path, eval_features_path, parameters['mfcc_wl'], parameters['mfcc_ws'], parameters['mfcc_nf'], parameters['mfcc_nceps'], parameters['mfcc_fmin'], parameters['mfcc_fmax'], parameters['mfcc_d_w'], parameters['mfcc_pre'], parameters['mfcc_mel'])
#
# #3. Read training features to array
# logger.info('Load train features')
# #analyze.recursive_test_wavs_for_nan(sound_path)
# #analyze.recursive_test_for_nan(train_features_path)
#
# train_features = analyze.recursive_load_mfcc_files(train_features_path)
#
# #4. array is converted to multidimensional ndarray
# logger.info('Convert train features array')
# train_features = numpy.vstack(train_features)
#
# #5. Clustering the train data using k-means and save result
# logger.info('Train k-means')
#
# kmeans = None
#
# if os.path.isfile(kmeans_file):
#     logger.info('K-Means file exists (' + kmeans_file + '). No new K-Means is generated.')
#     kmeans_hdf5 = bob.io.HDF5File(kmeans_file)
#     kmeans = bob.machine.KMeansMachine(kmeans_hdf5)
#     logger.info('K-Means file loaded')
# else:
#     kmeans = analyze.train_kmeans_machine(train_features, parameters['kmeans_num_gauss'], parameters['kmeans_dim'])
#     logger.info('save K-Means to file: ' + kmeans_file)
#     kmeans.save(bob.io.HDF5File(kmeans_file, 'w'))
#
# #analyze.test_array_for_nan(kmeans.means)
# #6. Create the universal background model
#
#
# #7. Train ubm-gmm and save result
# logger.info('Train UBM-GMM')
#
# ubm = None
#
# #If UBM file exists, open it
# if os.path.isfile(ubm_file):
#     logger.info('UBM file exists (' + ubm_file + '). No new UBM is generated.')
#     ubm_hdf5 = bob.io.HDF5File(ubm_file)
#     ubm = bob.machine.GMMMachine(ubm_hdf5)
#     logger.info('UBM file loaded')
# else:
#     logger.info('No UBM file found (' + ubm_file + '). New UBM is generated.')
#     ubm = analyze.train_ubm_gmm_with_features(kmeans, train_features, parameters['ubm_gmm_num_gauss'], parameters['ubm_gmm_dim'], parameters['ubm_convergence_threshold'], parameters['ubm_max_iterations'])
# #Save ubm
#     logger.info('Save UBM to file: ' + ubm_file)
#     ubm.save(bob.io.HDF5File(ubm_file, "w"))
#
#
# logger.info('Compute GMM sufficient statistics for both training and eval sets')
# #8. Compute GMM sufficient statistics for both training and eval sets
# dim = train_features.shape[1]
# logger.info('Train features dimension = ' + str(dim))
#
# if os.path.exists(gmm_stats_train_path):
#     logger.info('GMM train stats path exists ('+ gmm_stats_train_path +') Skipping... ')
# else:
#     logger.info('Calculating GMM train stats')
#     analyze.compute_gmm_sufficient_statistics(ubm, train_features_path, gmm_stats_train_path)
#
# if os.path.exists(gmm_stats_eval_path):
#     logger.info('GMM evaluation stats path exists ('+ gmm_stats_eval_path +') Skipping... ')
# else:
#     logger.info('Calculating GMM evaluation stats')
#     analyze.compute_gmm_sufficient_statistics(ubm, eval_features_path, gmm_stats_eval_path)
#
# #9. Training TV and Sigma matrices
# logger.info('Training ivector machine')
# gmm_train_stats = analyze.recursive_load_gmm_stats(gmm_stats_train_path)
#
# ivec_machine = None
#
# if os.path.isfile(ivec_machine_file):
#     logger.info('Ivector machine file exists (' + ivec_machine_file + '). No new ivector machine is generated.')
#     tv_hdf5 = bob.io.HDF5File(ivec_machine_file)
#     ivec_machine = bob.machine.IVectorMachine(tv_hdf5)
#     logger.info('Ivector machine loaded')
# else:
#     logger.info('Generating ivector machine.')
#     ivec_machine = analyze.gen_ivec_machine(gmm_train_stats, ubm, parameters['ivec_machine_dim'], parameters['ivec_machine_variance_treshold'], parameters['ivec_trainer_max_iterations'], parameters['ivec_trainer_update_sigma'])
#     logger.info('Ivector machine generated.')
#     #save the TV matrix
#     logger.info('Saving ivector machine to ' + ivec_machine_file)
#     ivec_machine.save(bob.io.HDF5File(ivec_machine_file, 'w'))
#
# #10. Extract i-vectors of the eval set..."
#
# logger.info('Extract i-vectors of the evaluation set.')
# if os.path.exists(ivec_dir):
#     logger.info('Ivectors  for evaluation set found so no ivectors are calculated.')
# else:
#     logger.info('Calculating ivectors for evaluation set')
#     analyze.extract_i_vectors(gmm_stats_eval_path, ivec_dir, ivec_machine)
#
# logger.info('Generate score file for ivector comparisons')
# if os.path.exists(eval_ivector_score_file):
#     logger.info('Score file for ivector analysis already exists in '+eval_ivector_score_file+'. No new score file is generated.')
# else:
#     logger.info('Generating score file for ivector analysis')
#     files = analyze.recursive_find_all_files(ivec_dir, '.hdf5')
#     evaluate.gen_score_file(files, eval_ivector_score_file, ivec_dir)
#     logger.info('Score file generated to: ' + eval_ivector_score_file)
#
#
# logger.info('Train GMM Machine with Map adaptation')
# gmm_machine = None
#
# # if os.path.isfile(gmm_machine_file):
# #     logger.info('GMM Machine file exists (' + gmm_machine_file + '). Loading MAP GMM Machine from file instead of generating it.')
# #     gmm_machine_hdf5 = bob.io.HDF5File(gmm_machine_file)
# #     gmm_machine = bob.machine.GMMMachine(gmm_machine_hdf5)
# #     logger.info('MAP GMM Machine file loaded')
# # else:
# #     logger.info('Generating MAP GMM Machine.')
# #     gmm_machine = analyze.gen_MAP_GMM_machine(ubm, train_features, parameters['gmm_adapted_num_gauss'], parameters['gmm_adapted_dim'], parameters['map_gmm_relevance_factor'], parameters['map_gmm_convergence_threshold'], parameters['map_gmm_max_iterations'])
# #     logger.info('TV matrix generated.')
# #     #save the TV matrix
# #     logger.info('Saving MAP GMM Machine to ' + gmm_machine_file)
# #     gmm_machine.save(bob.io.HDF5File(gmm_machine_file, 'w'))
# #
# # logger.info('Run evaluation set through map gmm machine.')
# # if os.path.exists(map_gmm_dir):
# #     logger.info('map gmm path already exists.')
# # else:
# #     logger.info('Calculating map gmm for evaluation set')
# #     analyze.map_gmm_machine_analysis(eval_features_path, map_gmm_dir, gmm_machine)
# #
# # logger.info('Generate score file for gmm comparisons')
# #
# # if os.path.exists(eval_map_gmm_score_file):
# #     logger.info('Score file for map gmm analysis already exists in '+eval_map_gmm_score_file+'. No new score file is generated.')
# # else:
# #     logger.info('Generating score file for map_gmm analysis')
# #     files = analyze.recursive_find_all_files(ivec_dir, '.hdf5')
# # #    gen_score_file(files, eval_map_gmm_score_file)
# #     logger.info('Score file generated to: ' + eval_map_gmm_score_file)
#
#
# if os.path.exists(map_gmm_dir):
#     logger.info('map gmm path already exists.')
# else:
#     logger.info('Calculating map gmm for evaluation set')
#     train_feature_files = analyze.recursive_find_all_files(train_features_path, '.hdf5')
#     train_feat_files_dict = analyze.gen_filedict(train_feature_files)
#
#     map_gmm_trainer = analyze.gen_map_gmm_trainer(ubm, parameters['map_gmm_relevance_factor'], parameters['map_gmm_convergence_threshold'], parameters['map_gmm_max_iterations'])
#     utils.ensure_dir(map_gmm_dir)
#     analyze.gen_MAP_GMM_machines(ubm, train_feat_files_dict, map_gmm_trainer, map_gmm_dir)
#
# # eva_gmm_stats = analyze.recursive_load_gmm_stats(gmm_stats_eval_path)
# # class_gmms = analyze.recursive_load_gmm_machines(map_gmm_dir)
#
# logger.info('Generate score file for gmm comparisons')
#
# if os.path.exists(eval_map_gmm_score_file):
#     logger.info('Score file for map gmm analysis already exists in '+eval_map_gmm_score_file+'. No new score file is generated.')
# else:
#     logger.info('Generating score file for map_gmm analysis')
#     class_gmms_files = analyze.recursive_find_all_files(map_gmm_dir, '.hdf5')
#     eval_gmm_stats_files = analyze.recursive_find_all_files(gmm_stats_eval_path, '.hdf5')
#     analyze.calc_scores(class_gmms_files, eval_gmm_stats_files, bob.machine.linear_scoring, ubm, eval_map_gmm_score_file)
#
# #EVALUATE RESULTS
#
# print_evaluation_log_file(eval_log_file)
#
#
# logger.info('Evaluating results')
# logger.info('Evaluating ivector results')
# logger.info('Finding negatives and positives from score file')
# negatives, positives = evaluate.parse_scores_from_file(eval_ivector_score_file)
# evaluate.evaluate_score_file(negatives, positives,test_roc_eval_ivec_file, test_det_eval_ivec_file, eval_log_file)
# logger.info('Evaluating map adapted gmm results')
# logger.info('Finding negatives and positives from score file')
# negatives, positives = evaluate.parse_scores_from_file(eval_map_gmm_score_file)
# evaluate.evaluate_score_file(negatives, positives, test_roc_eval_map_gmm_file, test_det_eval_map_gmm_file, eval_log_file)
#
# total_end_time = time.clock()
#
# logger.info('Total time elapsed: ' + str(total_end_time - total_start_time))