#!/usr/bin/env python
import argparse
import logging
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('../')

from deepfold.data.utils.data_utils import FUNC_DICT, NAMESPACES
from deepfold.data.utils.ontology import Ontology
from deepfold.loss.custom_metrics import evaluate_annotations

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
parser.add_argument('--diamond-scores-file',
                    '-dsf',
                    default='data/test_diamond.res',
                    help='Diamond output')
parser.add_argument('--ontology-obo-file',
                    '-obo',
                    default='data/go.obo',
                    help='Ontology file')
parser.add_argument('--output_dir', '-o', default='./', help='output dir')


def get_diamond_scores(diamond_scores_file):
    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])
    return diamond_scores


def get_diamond_preds(train_df, test_df, diamond_scores):
    """index     proteins       accessions 63995  454399  ST1S3_DANRE Q7T2V2;
    36226  229770  LYAM1_MOUSE          P18337; 73638  523000 XYNB_PRUPE
    P83344; 3003    14698   ALKH_ECOLI  P0A955; P10177; 25019  151684
    GEPH_MOUSE  Q8BUV3; E9QKJ1;

    Return:
        [
            {'GO1': 0.3, GO2': 0.3},
            {'GO3': 0.3, GO4': 0.3},
        ]
    """

    # protein name to index
    protein_index = {}
    for i, row in enumerate(train_df.itertuples()):
        protein_index[row.proteins] = i

    # annotations
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    # compute diamond predictions
    diamond_preds = []
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            # 根据比对结果将相似序列的功能作为预测结果
            for p_id, score in sim_prots.items():
                allgos |= annotations[protein_index[p_id]]
                total_score += score
            allgos = list(sorted(allgos))
            # 得到所有注释的功能并集
            sim = np.zeros(len(allgos), dtype=np.float32)
            # 计算每个功能注释的得分
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in annotations[protein_index[p_id]]:
                        s += score
                sim[j] = s / total_score
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        # 返回的 diamond_preds 是一个列表，
        diamond_preds.append(annots)
    return diamond_preds


def evaluate_diamond(test_df, blast_preds, go_rels, ont):
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
            for go_id, score in blast_preds[i].items():
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


def plot_diamond_aupr(precisions, recalls, aupr, ont, save_path):
    plt.figure()
    plt.plot(recalls,
             precisions,
             color='darkorange',
             lw=2,
             label=f'AUPR curve (area = {aupr:0.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under the Precision-Recall curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path + ont + '_aupr.png')


def main(train_data_file,
         test_data_file,
         diamond_scores_file,
         go_obo_file,
         output_dir=None,
         onts=('bp', 'mf', 'cc')):

    go_rels = Ontology(go_obo_file, with_rels=True)

    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_df = pd.read_pickle(test_data_file)
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    diamond_scores = get_diamond_scores(diamond_scores_file)
    blast_preds = get_diamond_preds(train_df, test_df, diamond_scores)
    for ont in onts:
        logger.info(f'Evaluate the {ont} protein family')
        go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
        go_set.remove(FUNC_DICT[ont])

        precisions, recalls, aupr = evaluate_diamond(test_df, blast_preds,
                                                     go_rels, ont)
        plot_diamond_aupr(precisions, recalls, aupr, ont, output_dir)


if __name__ == '__main__':
    logger = logging.getLogger('')
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    args = parser.parse_args()

    main(args.train_data_file, args.test_data_file, args.diamond_scores_file,
         args.ontology_obo_file, args.output_dir)
