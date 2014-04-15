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
    scores = load_score_file(score_file)
    negatives, positives = parse_neg_and_pos_scores(scores)

    negatives = numpy.array(negatives)
    positives = numpy.array(positives)

    return negatives, positives


def gen_roc_curve(negatives, positives, roc_curve_file, npoints=100):
    bob.measure.plot.roc(negatives, positives, npoints, color=(0,0,0), linestyle='-', label='ROC')
    pyplot.xlabel('FRR (%)')
    pyplot.ylabel('FAR (%)')
    pyplot.grid(True)
    pyplot.savefig(roc_curve_file)

def gen_det_curve(negatives, positives, det_curve_file, npoints=100):
    bob.measure.plot.det(negatives, positives, npoints, color=(0,0,0), linestyle='-', label='DET')
    bob.measure.plot.det_axis([1, 99, 1, 99])
    #bob.measure.plot.det_axis([0.01, 40, 0.01, 40])
    pyplot.xlabel('FRR (%)')
    pyplot.ylabel('FAR (%)')
    pyplot.grid(True)
    pyplot.savefig(det_curve_file)



# T = 0.0 #Threshold: later we explain how one can calculate these
# FAR, FRR = bob.measure.farfrr(negatives, positives, T)
#
# T = bob.measure.eer_threshold(negatives, positives)
# print 'eer_threshold=' + str(T)
# T = bob.measure.min_hter_threshold(negatives, positives)
# print 'min_hter_threshold=' + str(T)






