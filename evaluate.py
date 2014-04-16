# coding=utf-8
import os
import re
import argparse
import logging
import bob
import numpy
import matplotlib
import utils
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot


logger = logging.getLogger(__name__)

__author__ = 'Timo Mikkil'

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('scores', help='Score file')
#
# args = parser.parse_args()
#
# score_file = os.path.abspath(args.scores)


def load_score_file(score_file):
    scores = []
    if not os.path.exists(score_file):
        return None
    file_obj = open(score_file, 'r')
    #reg_exp = '\",?\"?'
    reg_exp = '^\"(?P<name1>[^\"]+)\",\"(?P<name2>[^\"]+)\",\"(?P<score>[^\"]+)\"'
    #reg_exp = '^\".*/(?P<name1>[a-zA-Z_]+)[-_\d]*split_[\d]+\.hdf5\",\".*/(?P<name2>[a-zA-Z_]+)[-_\d]*split_[\d]+\.hdf5\",\"(?P<score>[^\"]*)\"'

    prog = re.compile(reg_exp)

    for line in file_obj:
        matchObj = prog.match(line)
        if matchObj is None:
            logger.error('no match ('+line+')')
            continue

        #print matchObj

        f1 = matchObj.group('name1')
        f2 = matchObj.group('name2')
        score = float(matchObj.group('score'))
        name1 = utils.get_bird_name_from_file_name(f1)
        name2 = utils.get_bird_name_from_file_name(f2)
        scores.append((name1, name2, score))

        #print 'name1=' + name1 + ', name2=' + name2 + ', score=' + str(score)

    return scores

def parse_neg_and_pos_scores(scores):
    negatives = []
    positives = []
    for comp in scores:
        name1, name2, score = comp

        if name1 == name2:
            positives.append(score)
        else:
            negatives.append(score)


        #print 'name1=' + name1 + ', name2=' + name2 + ', score=' + str(score)

    return negatives, positives

def parse_scores_from_file(score_file):
    logger.debug('parseing core file: ' + score_file)
    scores = load_score_file(score_file)
    negatives, positives = parse_neg_and_pos_scores(scores)

    negatives = numpy.array(negatives)
    positives = numpy.array(positives)
    logger.info('Found ' + str(len(negatives)) + ' negatives and ' + str(len(positives)) + ' positives')
    return negatives, positives


def gen_roc_curve(negatives, positives, roc_curve_file, npoints=100):
    pyplot.clf()
    bob.measure.plot.roc(negatives, positives, npoints, color=(0,0,0), linestyle='-', label='ROC')
    pyplot.xlabel('FRR (%)')
    pyplot.ylabel('FAR (%)')
    pyplot.grid(True)
    pyplot.savefig(roc_curve_file)

def gen_det_curve(negatives, positives, det_curve_file, npoints=100):
    pyplot.clf()
    bob.measure.plot.det(negatives, positives, npoints, color=(0,0,0), linestyle='-', label='DET')
    bob.measure.plot.det_axis([1, 99, 1, 99])
    #bob.measure.plot.det_axis([0.01, 40, 0.01, 40])
    pyplot.xlabel('FRR (%)')
    pyplot.ylabel('FAR (%)')
    pyplot.grid(True)
    pyplot.savefig(det_curve_file)



def evaluate_score_file(negatives, positives, roc_file, det_file, eval_log_file):

    # eer_rocch = bob.measure.eer_rocch(negatives, positives)
    # logger.info('eer_rocch=' + str(eer_rocch))
    # FAR_eer_rocch, FRR_eer_rocch = bob.measure.farfrr(negatives, positives, eer_rocch)
    # logger.info('FAR_eer_rocch=' + str(FAR_eer_rocch))
    # logger.info('FRR_eer_rocch=' + str(FRR_eer_rocch))
    # correct_negatives_eer_rocch = bob.measure.correctly_classified_negatives(negatives)

    eer_threshold = bob.measure.eer_threshold(negatives, positives)
    logger.info('eer_threshold=' + str(eer_threshold))
    FAR_eer_threshold, FRR_eer_threshold = bob.measure.farfrr(negatives, positives, eer_threshold)
    logger.info('FAR_eer_threshold=' + str(FAR_eer_threshold))
    logger.info('FRR_eer_threshold=' + str(FRR_eer_threshold))
    correct_negatives_FAR_eer_threshold = bob.measure.correctly_classified_negatives(negatives, FAR_eer_threshold).sum()
    correct_positives_FRR_eer_threshold = bob.measure.correctly_classified_positives(positives, FRR_eer_threshold).sum()

    # min_hter_treshold = bob.measure.min_hter_threshold(negatives, positives)
    # logger.info('min_hter_treshold=' + str(min_hter_treshold))
    # FAR_min_hter_treshold, FRR_min_hter_treshold = bob.measure.farfrr(negatives, positives, min_hter_treshold)
    # logger.info('FAR_min_hter_treshold=' + str(FAR_min_hter_treshold))
    # logger.info('FRR_min_hter_treshold=' + str(FRR_min_hter_treshold))
    # correct_negatives_min_hter_treshold = bob.measure.correctly_classified_negatives(negatives, min_hter_treshold)
    # correct_positives_min_hter_treshold = bob.measure.correctly_classified_positives(positives, min_hter_treshold)
    with open(eval_log_file, "a") as f:
        f.write('\n\nEvaluation')
        f.write('\nNegatives: ' + str(len(negatives)))
        f.write('\nPositives: ' + str(len(positives)))
        f.write('\nEER treshold:' + str(eer_threshold))
        f.write('\nFAR: ' + str(FAR_eer_threshold) + ' (' + str(correct_negatives_FAR_eer_threshold) + '/' +str(len(negatives))+')')
        f.write('\nFRR: ' + str(FRR_eer_threshold) + ' (' + str(correct_positives_FRR_eer_threshold) + '/' +str(len(positives))+')')



        # f.write('\nMin HTER treshold:' + str(min_hter_treshold))
        # f.write('FAR: ' + str(FAR_min_hter_treshold))
        # f.write('FRR: ' + str(FRR_min_hter_treshold))
        # f.write('Correctly classified negatives: ' + str(correct_negatives_min_hter_treshold))
        # f.write('Correctly classified positives: ' + str(correct_positives_min_hter_treshold))



    logger.info('Generating ROC curve to ' + roc_file)
    gen_roc_curve(negatives, positives, roc_file)

    logger.info('Generating DET curve to ' + det_file)
    gen_det_curve(negatives, positives, det_file)

def print_evaluation_log_file(filename, params):
    with open(filename, "w") as f:
        f.write('Parameters used\n')
        for key in params.keys():
            f.write('' + key + ': ' + str(params[key])+'\n')


        # f.write('Parameters used')
        # f.write('\nparameters['kmeans_num_gauss']=' + str(parameters['kmeans_num_gauss']))
        # f.write('\nparameters['kmeans_dim']=' + str(parameters['kmeans_dim']))
        #
        # #GMM stats parameters
        # f.write('\n\nGMM stats parameters:')
        # f.write('\nparameters['gmm_stat_num_gauss']=' + str(parameters['gmm_stat_num_gauss']))
        # f.write('\nparameters['gmm_stat_dim']=' + str(parameters['gmm_stat_dim']))
        #
        # #Ivector machine parameters
        # f.write('\n\nIvector machine parameters:')
        # f.write('\nparameters['ivec_machine_dim']=' + str(parameters['ivec_machine_dim']))
        # f.write('\nparameters['ivec_machine_variance_treshold']=' + str(parameters['ivec_machine_variance_treshold']))
        #
        # #Ivector trainer parameters
        # f.write('\n\nIvector trainer parameters:')
        # f.write('\nparameters['ivec_trainer_max_iterations']=' + str(parameters['ivec_trainer_max_iterations']))
        # f.write('\nparameters['ivec_trainer_update_sigma']=' + str(parameters['ivec_trainer_update_sigma']))
        #
        # #Parameters for map gmm trainer
        # f.write('\n\nMap gmm trainer parameters:')
        # f.write('\nparameters['map_gmm_relevance_factor']=' + str(parameters['map_gmm_relevance_factor']))
        # f.write('\nparameters['map_gmm_convergence_threshold']=' + str(parameters['map_gmm_convergence_threshold']))
        # f.write('\nparameters['map_gmm_max_iterations']=' + str(parameters['map_gmm_max_iterations']))
        #
        # #Parameters for map adapted gmm machine
        # f.write('\n\nMap adapted gmm machine parameters:')
        # f.write('\nparameters['gmm_adapted_num_gauss']=' + str(parameters['gmm_adapted_num_gauss']))
        # f.write('\nparameters['gmm_adapted_dim']=' + str(parameters['gmm_adapted_dim']))
        #
        # f.write('\n\nUBM parameters:')
        # f.write('\nparameters['ubm_gmm_num_gauss']=' + str(parameters['ubm_gmm_num_gauss']))
        # f.write('\nparameters['ubm_gmm_dim']=' + str(parameters['ubm_gmm_dim']))
        # f.write('\nparameters['ubm_convergence_threshold']=' + str(parameters['ubm_convergence_threshold']))
        # f.write('\nparameters['ubm_max_iterations']=' + str(parameters['ubm_max_iterations']))


# T = 0.0 #Threshold: later we explain how one can calculate these
# FAR, FRR = bob.measure.farfrr(negatives, positives, T)
#
# T = bob.measure.eer_threshold(negatives, positives)
# print 'eer_threshold=' + str(T)
# T = bob.measure.min_hter_threshold(negatives, positives)
# print 'min_hter_threshold=' + str(T)






