#!/usr/bin/env python

import argparse
import logging
import math
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from deepfold.data.utils.data_utils import FUNC_DICT, NAMESPACES
from deepfold.data.utils.ontology import Ontology
from deepfold.loss.custom_metrics import evaluate_annotations

sys.path.append('../')

parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--train-data-file',
                    '-trdf',
                    default='data/train_data.pkl',
                    help='Data file with training features')
parser.add_argument('--test-data-file',
                    '-tsdf',
                    default='data/test_data.pkl',
                    help='Data file with test')
parser.add_argument(
    '--terms-file',
    '-tf',
    default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
parser.add_argument('--diamond-scores-file',
                    '-dsf',
                    default='data/test_diamond.res',
                    help='Diamond output')
parser.add_argument('--ontology-obo-file',
                    '-obo',
                    default='data/go.obo',
                    help='Ontology file')
parser.add_argument('--output_dir', '-o', default='./', help='output dir')

alphas = {NAMESPACES['mf']: 0, NAMESPACES['bp']: 0, NAMESPACES['cc']: 0}


def get_model_preds(test_df, terms):
    model_preds = []
    for i, row in enumerate(test_df.itertuples()):
        annots_dict = {}
        for j, score in enumerate(row.preds):
            go_id = terms[j]
            annots_dict[go_id] = score
        model_preds.append(annots_dict)

    return model_preds


def evaluate_model_prediction(test_df, model_preds, go_rels, ont, logger=None):
    fmax = 0.0
    tmax = 0.0
    smin = 1000.0

    precisions = []
    recalls = []
    # test_annotations
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))

    # go set
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

    # labels
    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))

    for t in range(0, 101, 10):
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            for go_id, score in model_preds[i].items():
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)

        # Filter classes
        preds = list(
            map(lambda x: set(filter(lambda y: y in go_set, x)), preds))

        fscore, prec, rec, s, _, _, _, _ = evaluate_annotations(
            go_rels, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        logger.info(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s

    logger.info(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    logger.info(f'AUPR: {aupr:0.3f}')
    return precisions, recalls, aupr


def main(train_data_file, test_data_file, terms_file, diamond_scores_file, ont,
         alpha):

    go_rels = Ontology('data/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    print('Length of test set: ' + str(len(test_df)))

    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    # DeepGOPlus
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = test_df['prop_annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    # print(len(go_set))
    deep_preds = []
    # alphas = {NAMESPACES['mf']: 0.55, NAMESPACES['bp']: 0.59, NAMESPACES['cc']: 0.46}
    alphas = {NAMESPACES['mf']: 0, NAMESPACES['bp']: 0, NAMESPACES['cc']: 0}

    for i, row in enumerate(test_df.itertuples()):
        annots_dict = blast_preds[i].copy()
        for go_id in annots_dict:
            annots_dict[go_id] *= alphas[go_rels.get_namespace(go_id)]
        for j, score in enumerate(row.preds):
            go_id = terms[j]
            score *= 1 - alphas[go_rels.get_namespace(go_id)]
            if go_id in annots_dict:
                annots_dict[go_id] += score
            else:
                annots_dict[go_id] = score
        deep_preds.append(annots_dict)


if __name__ == '__main__':
    main()
