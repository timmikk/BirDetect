# coding=utf-8
import logging
import numpy
import bob
import os
import time

from bunch import Bunch

import analyze
import evaluate


class ivector_worker:
    logger = logging.getLogger(__name__)
    paths = Bunch()

    def __init__(self, paths, params):
        self.paths = paths
        self.params = params

    def gen_score_file(self, files, score_file, dest_dir):
        #Load all ivec file names to array
        #Compare files to each other and write to score file
        self.logger.debug('Generating score file to ' + score_file)
        f = open(score_file, 'w')

        for f1 in files:
            for f2 in files:
                f1_loaded = bob.io.load(f1)
                f2_loaded = bob.io.load(f2)
                f1_loaded = numpy.linalg.norm(f1_loaded)
                f2_loaded = numpy.linalg.norm(f2_loaded)
                score = numpy.dot(f1_loaded, f2_loaded)

                #score = utils.cosine_score(f1_loaded, f2_loaded)
                f.write(
                    '\"' + f1[len(dest_dir) + 1:] + '\",\"' + f2[len(dest_dir) + 1:] + '\",\"' + str(score) + '\"\n')

        f.close()

    def run(self):

        total_start_time = time.clock()
        #1. Extract and save features from training files
        if os.path.exists(self.paths.features_train):
            self.logger.warn(
                'Features are not extracted for train data as folder already exists(' + self.paths.features_train + ').')
        else:
            self.logger.info('Extracting mfcc features from train data')
            self.logger.debug('From ' + self.paths.sounds_train + ' to ' + self.paths.features_train)
            analyze.recursively_extract_features(self.paths.sounds_train, self.paths.features_train,
                                                 self.params.mfcc_wl,
                                                 self.params.mfcc_ws, self.params.mfcc_nf, self.params.mfcc_nceps,
                                                 self.params.mfcc_fmin, self.params.mfcc_fmax, self.params.mfcc_d_w,
                                                 self.params.mfcc_pre, self.params.mfcc_mel)

        #2. Extract and save features from evaluation files
        if os.path.exists(self.paths.features_eval):
            self.logger.warn(
                'Features are not extracted for evaluation data as folder already exists (' + self.paths.features_eval + ').')
        else:
            self.logger.info('Extracting mfcc features from evaluation data')
            self.logger.debug('From ' + self.paths.sounds_eval + ' to ' + self.paths.features_eval)
            analyze.recursively_extract_features(self.paths.sounds_eval, self.paths.features_eval, self.params.mfcc_wl,
                                                 self.params.mfcc_ws, self.params.mfcc_nf, self.params.mfcc_nceps,
                                                 self.params.mfcc_fmin, self.params.mfcc_fmax, self.params.mfcc_d_w,
                                                 self.params.mfcc_pre, self.params.mfcc_mel)

        #3. Read training features to array
        self.logger.info('Load train features')
        #analyze.recursive_test_wavs_for_nan(self.paths.sound)
        #analyze.recursive_test_for_nan(self.paths.features_train)

        features_train = analyze.recursive_load_mfcc_files(self.paths.features_train)

        #4. array is converted to multidimensional ndarray
        self.logger.info('Convert train features array')
        training_features_set = numpy.vstack(features_train)
        features_train = None

        #5. Clustering the train data using k-means and save result
        self.logger.info('Train k-means')

        kmeans = None

        if os.path.isfile(self.paths.kmeans_file):
            self.logger.info('K-Means file exists (' + self.paths.kmeans_file + '). No new K-Means is generated.')
            kmeans_hdf5 = bob.io.HDF5File(self.paths.kmeans_file)
            kmeans = bob.machine.KMeansMachine(kmeans_hdf5)
            self.logger.info('K-Means file loaded')
        else:
            #kmeans = analyze.train_kmeans_machine(features_train, self.params.kmeans_num_gauss, self.params.kmeans_dim)
            kmeans = analyze.gen_kmeans(training_features_set, self.params.number_of_gaussians)
            self.logger.info('save K-Means to file: ' + self.paths.kmeans_file)
            kmeans.save(bob.io.HDF5File(self.paths.kmeans_file, 'w'))

        #analyze.test_array_for_nan(kmeans.means)
        #6. Create the universal background model


        #7. Train ubm-gmm and save result
        self.logger.info('Train UBM-GMM')

        ubm = None

        #If UBM file exists, open it
        if os.path.isfile(self.paths.ubm_file):
            self.logger.info('UBM file exists (' + self.paths.ubm_file + '). No new UBM is generated.')
            ubm_hdf5 = bob.io.HDF5File(self.paths.ubm_file)
            ubm = bob.machine.GMMMachine(ubm_hdf5)
            self.logger.info('UBM file loaded')
        else:
            self.logger.info('No UBM file found (' + self.paths.ubm_file + '). New UBM is generated.')
            #ubm = analyze.train_ubm_gmm_with_features(kmeans, features_train, self.params.ubm_gmm_num_gauss,
            #                                          self.params.ubm_gmm_dim, self.params.ubm_convergence_threshold,
            #                                          self.params.ubm_max_iterations)
            ubm = analyze.gen_ubm(kmeans, training_features_set, self.params.ubm_trainer_convergence_threshold,
                                  self.params.ubm_trainer_max_iterations)
            #Save ubm
            self.logger.info('Save UBM to file: ' + self.paths.ubm_file)
            ubm.save(bob.io.HDF5File(self.paths.ubm_file, "w"))

        training_features_set = None
        kmeans = None

        self.logger.info('Compute GMM sufficient statistics for both training and eval sets')
        #8. Compute GMM sufficient statistics for both training and eval sets

        self.logger.info('Compute GMM sufficient statistics for both training and eval sets')
        #8. Compute GMM sufficient statistics for both training and eval sets

        if os.path.exists(self.paths.gmm_stats_train):
            self.logger.info('GMM train stats path exists (' + self.paths.gmm_stats_train + ') Skipping... ')
        else:
            self.logger.info('Calculating GMM train stats')
            analyze.compute_gmm_sufficient_statistics(ubm, self.paths.features_train, self.paths.gmm_stats_train)

        if os.path.exists(self.paths.gmm_stats_eval):
            self.logger.info('GMM evaluation stats path exists (' + self.paths.gmm_stats_eval + ') Skipping... ')
        else:
            self.logger.info('Calculating GMM evaluation stats')
            analyze.compute_gmm_sufficient_statistics(ubm, self.paths.features_eval, self.paths.gmm_stats_eval)

        #9. Training TV and Sigma matrices
        self.logger.info('Training ivector machine')
        gmm_train_stats = analyze.recursive_load_gmm_stats(self.paths.gmm_stats_train)

        ivec_machine = None

        if os.path.isfile(self.paths.ivec_machine_file):
            self.logger.info(
                'Ivector machine file exists (' + self.paths.ivec_machine_file + '). No new ivector machine is generated.')
            tv_hdf5 = bob.io.HDF5File(self.paths.ivec_machine_file)
            ivec_machine = bob.machine.IVectorMachine(tv_hdf5)
            self.logger.info('Ivector machine loaded')
        else:
            self.logger.info('Generating ivector machine.')
            ivec_machine = analyze.gen_ivec_machine(gmm_train_stats, ubm, self.params.ivec_machine_dim,
                                                    self.params.ivec_machine_variance_treshold,
                                                    self.params.ivec_trainer_max_iterations,
                                                    self.params.ivec_trainer_update_sigma)
            self.logger.info('Ivector machine generated.')
            #save the TV matrix
            self.logger.info('Saving ivector machine to ' + self.paths.ivec_machine_file)
            ivec_machine.save(bob.io.HDF5File(self.paths.ivec_machine_file, 'w'))

        #10. Extract i-vectors of the eval set..."

        self.logger.info('Extract i-vectors of the evaluation set.')
        if os.path.exists(self.paths.ivectors_eval):
            self.logger.info('Ivectors  for evaluation set found so no ivectors are calculated.')
        else:
            self.logger.info('Calculating ivectors for evaluation set')
            analyze.extract_i_vectors(self.paths.gmm_stats_eval, self.paths.ivectors_eval, ivec_machine)

        self.logger.info('Generate score file for ivector comparisons')
        if os.path.exists(self.paths.scores_ivec):
            self.logger.info(
                'Score file for ivector analysis already exists in ' + self.paths.scores_ivec + '. No new score file is generated.')
        else:
            self.logger.info('Generating score file for ivector analysis')
            files = analyze.recursive_find_all_files(self.paths.ivectors_eval, '.hdf5')
            self.gen_score_file(files, self.paths.scores_ivec, self.paths.ivectors_eval)
            self.logger.info('Score file generated to: ' + self.paths.scores_ivec)

        #EVALUATE RESULTS

        evaluate.print_evaluation_log_file(self.paths.eval_ivec_log, self.params)

        self.logger.info('Evaluating results')
        self.logger.info('Evaluating ivector results')
        self.logger.info('Finding negatives and positives from score file')
        negatives, positives = evaluate.parse_scores_from_file(self.paths.scores_ivec)
        evaluate.evaluate_score_file(negatives, positives, self.paths.eval_roc_ivec, self.paths.eval_det_ivec,
                                     self.paths.eval_ivec_log)

        total_end_time = time.clock()

        self.logger.info('Total time elapsed: ' + str(total_end_time - total_start_time))