# coding=utf-8
import os
import logging
import logging.config
import argparse
import yaml

from bunch import Bunch

import ivector
import ubmgmm


__author__ = 'Timo Mikkilä'

logger = logging.getLogger(__name__)

parameters = {}


def load_parameters(param_file):
    if not os.path.isfile(param_file):
        logger.error('Error no such parameter file:' + param_file)
        exit(128)
    f = open(param_file)
    global parameters
    new_params = yaml.load(f)
    parameters = dict(parameters.items() + new_params.items())


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run_mode', help='What to do', choices=['ivector', 'ubm-gmm', 'all'], default='all')
parser.add_argument('--num_gauss', help='Number of gaussian used in calculations', type=int)
parser.add_argument('--snd_path', help='Location of wav files. Must contain train and eval folders', default='snd')
parser.add_argument('--dest_path', help='Location where all data is written to', default='dest')
parser.add_argument('--params_file', help='Parameter file')

args = parser.parse_args()

paths = Bunch()

paths.dest = os.path.abspath(args.dest_path)
paths.sound = os.path.abspath(args.snd_path)
paths.sounds_train = os.path.join(paths.sound, 'train')
paths.sounds_eval = os.path.join(paths.sound, 'eval')
paths.features = os.path.join(paths.dest, 'features')
paths.features_train = os.path.join(paths.features, 'train')
paths.features_eval = os.path.join(paths.features, 'eval')
paths.kmeans_file = os.path.join(paths.dest, 'kmeans.hdf5')
paths.ubm_file = os.path.join(paths.dest, 'ubm.hdf5')
paths.gmms = os.path.join(paths.dest, 'gmm_stats')
paths.gmm_stats_train = os.path.join(paths.gmms, 'train')
paths.gmm_stats_eval = os.path.join(paths.gmms, 'eval')
paths.ivec_machine_file = os.path.join(paths.dest, 'ivec_machine.hdf5')
paths.ivectors_eval = os.path.join(paths.dest, 'ivectors')
paths.class_gmms = os.path.join(paths.dest, 'class_gmms')
paths.scores_ivec = os.path.join(paths.dest, 'scores-ivec.txt')
paths.scores_ubm_gmm = os.path.join(paths.dest, 'scores-ubm-gmm.txt')
paths.eval_roc_ubm_gmm = os.path.join(paths.dest, 'roc-ubm-gmm.png')
paths.eval_det_ubm_gmm = os.path.join(paths.dest, 'det-ubm-gmm.png')
paths.eval_roc_ivec = os.path.join(paths.dest, 'roc-ivec.png')
paths.eval_det_ivec = os.path.join(paths.dest, 'det-ivec.png')
paths.eval_ivec_log = os.path.join(paths.dest, 'eval-ivec.txt')
paths.eval_ubm_gmm_log = os.path.join(paths.dest, 'eval-ubm-gmm.txt')
paths.conf_default_file = os.path.join(os.path.dirname(__file__), 'default.cfg')

log_file_locator = lambda *x: os.path.join(paths.dest, *x)

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

logging.config.dictConfig(LOGGING)

logger.info(str(paths))

logger.info('Loading parameters from ' + paths.conf_default_file)

#Load default parameters
load_parameters(paths.conf_default_file)

if args.params_file is not None:
    logger.info('Loading user defined parameters from: ' + str(args.params_file))
    load_parameters(args.params_file)

params = Bunch(parameters)
logger.info('Parameters loaded: ' + str(params))

if (args.num_gauss is not None):
    logger.warning(
        'Overriding number of gaussians (' + str(params.number_of_gaussians) + ' -> ' + str(args.num_gauss) + ')')
    params.number_of_gaussians = args.num_gauss

if (args.run_mode is not None):
    logger.warning('Overriding run mode (' + str(params.run_mode) + ' -> ' + str(args.run_mode) + ')')
    params.run_mode = args.run_mode

if params.run_mode == 'ubm-gmm' or params.run_mode == 'all':
    worker = ubmgmm.ubm_gmm_worker(paths, params)
    try:
        worker.run()
    except:
        logging.exception('Something went wrong')
if params.run_mode == 'ivector' or params.run_mode == 'all':
    worker = ivector.ivector_worker(paths, params)
    try:
        worker.run()
    except:
        logging.exception('Something went wrong')
