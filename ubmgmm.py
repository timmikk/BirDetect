# coding=utf-8
import logging
import numpy
import bob
import os
import time

from bunch import Bunch

import analyze
import evaluate
import utils

__author__ = 'Timo Mikkilä'

class ubm_gmm_worker:
    logger = logging.getLogger(__name__)
    paths = Bunch()
    def __init__(self, paths, params):
        self.paths = paths
        self.params = params

    def calc_scores(self, class_gmms_files, probe_stats_files, distance_function, ubm, score_file):
        self.logger.debug('calc_scores')
        f = open(score_file, 'w')

        class_file_num = len(class_gmms_files)
        current_class_file_num = 0
        for class_gmm_file in class_gmms_files:
            current_class_file_num += 1
            self.logger.info('Calculating sores ' + str(current_class_file_num) + '/' + str(class_file_num) +  ' for class file ' + class_gmm_file)
            for stats_file in probe_stats_files:
                #self.logger.debug('calculating sore. Class file: ' + class_gmm_file + ' stats file: + stats_file')
                class_gmm = analyze.load_gmm_machine_file(class_gmm_file)
                stats = analyze.load_gmm_stats_file(stats_file)
                score = distance_function([class_gmm], ubm, [stats])[0,0]

                f.write('\"'+os.path.basename(class_gmm_file)+ '\",\"'+os.path.basename(stats_file)+'\",\"' + str(score) + '\"\n')
        f.close()

    def calc_scores_fast(self, class_gmms_files, probe_stats_files, distance_function, ubm, score_file):
        self.logger.debug('calc_scores')
        # f = open(score_file, 'w')

        #load class files
        glass_gmms = []
        for class_gmm_file in class_gmms_files:
            glass_gmms.append(analyze.load_gmm_machine_file(class_gmm_file))

        #load stats files
        gmm_stats = []
        for stats_file in probe_stats_files:
            gmm_stats.append(analyze.load_gmm_stats_file(stats_file))

        score = distance_function(glass_gmms, ubm, gmm_stats)
        self.logger.info('done')

        f = open(score_file, 'w')
        for class_index, class_file in enumerate(class_gmms_files):
            for stats_index, stats_file in enumerate(probe_stats_files):
                f.write('\"'+os.path.basename(class_file)+ '\",\"'+os.path.basename(stats_file)+'\",\"' + str(score[class_index][stats_index]) + '\"\n')

        # class_file_num = len(class_gmms_files)
        # current_class_file_num = 0
        # for class_gmm_file in class_gmms_files:
        #     current_class_file_num += 1
        #     self.logger.info('Calculating sores ' + str(current_class_file_num) + '/' + str(class_file_num) +  ' for class file ' + class_gmm_file)
        #     for stats_file in probe_stats_files:
        #         #self.logger.debug('calculating sore. Class file: ' + class_gmm_file + ' stats file: + stats_file')
        #         class_gmm = analyze.load_gmm_machine_file(class_gmm_file)
        #         stats = analyze.load_gmm_stats_file(stats_file)
        #         score = distance_function([class_gmm], ubm, [stats])[0,0]
        #
        #         f.write('\"'+os.path.basename(class_gmm_file)+ '\",\"'+os.path.basename(stats_file)+'\",\"' + str(score) + '\"\n')
        f.close()

    def run(self):
        total_start_time = time.clock()
        #1. Extract and save features from training files
        if os.path.exists(self.paths.features_train):
            self.logger.warn('Features are not extracted for train data as folder already exists(' + self.paths.features_train + ').')
        else:
            self.logger.info('Extracting mfcc features from train data')
            self.logger.debug('From ' + self.paths.sounds_train + ' to ' + self.paths.features_train)
            analyze.recursively_extract_features(self.paths.sounds_train, self.paths.features_train, self.params.mfcc_wl,
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
            ubm = analyze.gen_ubm(kmeans, training_features_set, self.params.ubm_trainer_convergence_threshold, self.params.ubm_trainer_max_iterations)
            #Save ubm
            self.logger.info('Save UBM to file: ' + self.paths.ubm_file)
            ubm.save(bob.io.HDF5File(self.paths.ubm_file, "w"))

        self.logger.info('Compute GMM sufficient statistics for both training and eval sets')
        #8. Compute GMM sufficient statistics for both training and eval sets

        features_train = None
        training_features_set = None
        kmeans = None

        if os.path.exists(self.paths.class_gmms):
            self.logger.info('map gmm path already exists.')
        else:
            self.logger.info('Calculating map gmm for evaluation set')
            train_feature_files = analyze.recursive_find_all_files(self.paths.features_train, '.hdf5')
            train_feat_files_dict = analyze.gen_filedict(train_feature_files)

            map_gmm_trainer = analyze.gen_map_gmm_trainer(ubm, self.params.map_gmm_relevance_factor,
                                                          self.params.map_gmm_convergence_threshold,
                                                          self.params.map_gmm_max_iterations)
            utils.ensure_dir(self.paths.class_gmms)
            analyze.gen_MAP_GMM_machines(ubm, train_feat_files_dict, map_gmm_trainer, self.paths.class_gmms)


        #Gen gmm statistics from evaluation set
        if os.path.exists(self.paths.gmm_stats_eval):
            self.logger.info('GMM evaluation stats path exists (' + self.paths.gmm_stats_eval + ') Skipping... ')
        else:
            self.logger.info('Calculating GMM evaluation stats')
            analyze.compute_gmm_sufficient_statistics(ubm, self.paths.features_eval, self.paths.gmm_stats_eval)

        # eva_gmm_stats = analyze.recursive_load_gmm_stats(self.paths.gmm_stats_eval)
        # class_gmms = analyze.recursive_load_gmm_machines(class_gmms)

        self.logger.info('Generate score file for gmm comparisons')

        if os.path.exists(self.paths.scores_ubm_gmm):
            self.logger.info(
                'Score file for map gmm analysis already exists in ' + self.paths.scores_ubm_gmm + '. No new score file is generated.')
        else:
            self.logger.info('Generating score file for map_gmm analysis')
            class_gmms_files = analyze.recursive_find_all_files(self.paths.class_gmms, '.hdf5')
            eval_gmm_stats_files = analyze.recursive_find_all_files(self.paths.gmm_stats_eval, '.hdf5')
            self.calc_scores_fast(class_gmms_files, eval_gmm_stats_files, bob.machine.linear_scoring, ubm,
                                self.paths.scores_ubm_gmm)


        #TEST


        # self.logger.info('Generating score file for map_gmm analysis')
        # class_gmms_files = analyze.recursive_find_all_files(self.paths.class_gmms, '.hdf5')
        # eval_gmm_stats_files = analyze.recursive_find_all_files(self.paths.gmm_stats_eval, '.hdf5')
        # self.logger.info('Start test run 1')
        # start_time = time.clock()
        # self.calc_scores(class_gmms_files, eval_gmm_stats_files, bob.machine.linear_scoring, ubm,
        #                     self.paths.scores_ubm_gmm)
        # end_time = time.clock()
        # self.logger.info('End test run 1. Time: ' + str(end_time - start_time))
        #
        # self.logger.info('Start test run 2')
        # start_time = time.clock()
        # #self.calc_scores2(class_gmms_files, eval_gmm_stats_files, bob.machine.linear_scoring, ubm,
        # #                    self.paths.scores_ubm_gmm)
        # end_time = time.clock()
        # self.logger.info('End test run 2. Time: ' + str(end_time - start_time))
        #TEST END





        #EVALUATE RESULTS

        evaluate.print_evaluation_log_file(self.paths.eval_ubm_gmm_log, self.params)

        self.logger.info('Evaluating results')
        self.logger.info('Evaluating map adapted gmm results')
        self.logger.info('Finding negatives and positives from score file')
        negatives, positives = evaluate.parse_scores_from_file(self.paths.scores_ubm_gmm)
        evaluate.evaluate_score_file(negatives, positives, self.paths.eval_roc_ubm_gmm, self.paths.eval_det_ubm_gmm, self.paths.eval_ubm_gmm_log)

        total_end_time = time.clock()

        self.logger.info('Total time elapsed: ' + str(total_end_time - total_start_time))

