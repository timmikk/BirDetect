# coding=utf-8
import os
import re
import logging
import bob
import numpy

import matplotlib

import utils


__author__ = 'Timo Mikkilä'

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot


logger = logging.getLogger(__name__)

__author__ = 'Timo Mikkil'



def load_score_file(score_file):
    scores = []
    if not os.path.exists(score_file):
        return None
    file_obj = open(score_file, 'r')
    reg_exp = '^\"(?P<name1>[^\"]+)\",\"(?P<name2>[^\"]+)\",\"(?P<score>[^\"]+)\"'

    prog = re.compile(reg_exp)

    for line in file_obj:
        matchObj = prog.match(line)
        if matchObj is None:
            logger.error('no match (' + line + ')')
            continue

        f1 = matchObj.group('name1')
        f2 = matchObj.group('name2')
        score = float(matchObj.group('score'))
        name1 = utils.get_bird_name_from_file_name(f1)
        name2 = utils.get_bird_name_from_file_name(f2)
        scores.append((name1, name2, score))

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
    bob.measure.plot.roc(negatives, positives, npoints, color=(0, 0, 0), linestyle='-', label='ROC')
    pyplot.xlabel('FRR (%)')
    pyplot.ylabel('FAR (%)')
    pyplot.grid(True)
    pyplot.savefig(roc_curve_file)


def gen_det_curve(negatives, positives, det_curve_file, npoints=100):
    pyplot.clf()
    bob.measure.plot.det(negatives, positives, npoints, color=(0, 0, 0), linestyle='-', label='DET')
    bob.measure.plot.det_axis([1, 99, 1, 99])
    #bob.measure.plot.det_axis([0.01, 40, 0.01, 40])
    pyplot.xlabel('FRR (%)')
    pyplot.ylabel('FAR (%)')
    pyplot.grid(True)
    pyplot.savefig(det_curve_file)


def evaluate_score_file(negatives, positives, roc_file, det_file, eval_log_file):

    eer_threshold = bob.measure.eer_threshold(negatives, positives)
    logger.info('eer_threshold=' + str(eer_threshold))
    FAR_eer_threshold, FRR_eer_threshold = bob.measure.farfrr(negatives, positives, eer_threshold)
    logger.info('FAR_eer_threshold=' + str(FAR_eer_threshold))
    logger.info('FRR_eer_threshold=' + str(FRR_eer_threshold))
    correct_negatives_FAR_eer_threshold = bob.measure.correctly_classified_negatives(negatives, FAR_eer_threshold).sum()
    correct_positives_FRR_eer_threshold = bob.measure.correctly_classified_positives(positives, FRR_eer_threshold).sum()

    with open(eval_log_file, "a") as f:
        f.write('\n\nEvaluation')
        f.write('\nNegatives: ' + str(len(negatives)))
        f.write('\nPositives: ' + str(len(positives)))
        f.write('\nEER treshold:' + str(eer_threshold))
        f.write('\nFAR: ' + str(FAR_eer_threshold) + ' (' + str(correct_negatives_FAR_eer_threshold) + '/' + str(
            len(negatives)) + ')')
        f.write('\nFRR: ' + str(FRR_eer_threshold) + ' (' + str(correct_positives_FRR_eer_threshold) + '/' + str(
            len(positives)) + ')')


    logger.info('Generating ROC curve to ' + roc_file)
    gen_roc_curve(negatives, positives, roc_file)

    logger.info('Generating DET curve to ' + det_file)
    gen_det_curve(negatives, positives, det_file)


def print_evaluation_log_file(filename, params):
    with open(filename, "w") as f:
        f.write('Parameters used\n')
        for key in params.keys():
            f.write('' + key + ': ' + str(params[key]) + '\n')





