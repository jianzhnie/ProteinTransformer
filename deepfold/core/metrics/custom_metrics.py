import math

import numpy as np
from sklearn import metrics
from sklearn.metrics import (auc, average_precision_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.utils import resample


# functions for evaluation
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_auc_score(labels, preds, average='macro'):
    # ROC AUC score
    ii = np.where(np.sum(labels, 0) > 0)[0]
    avg_rocauc = roc_auc_score(labels[:, ii], preds[:, ii], average=average)
    return avg_rocauc


def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def fmax(Ytrue, Ypred, nrThresholds):
    """get the minimum normalized semantic distance.

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        nrThresholds: the number of thresholds to check.

    OUTPUT:
        the minimum nsd that was achieved at the evaluated thresholds
    """
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ff = np.zeros(thresholds.shape)
    pr = np.zeros(thresholds.shape)
    rc = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        pr[i], rc[i], ff[i], _ = precision_recall_fscore_support(
            Ytrue, (Ypred >= t).astype(int), average='samples')

    return np.max(ff)


def smin(Ytrue, Ypred, termIC, nrThresholds):
    """get the minimum normalized semantic distance.

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        termIC: output of ic function above
        nrThresholds: the number of thresholds to check.

    OUTPUT:
        the minimum nsd that was achieved at the evaluated thresholds
    """

    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ss = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        ss[i] = normalizedSemanticDistance(Ytrue, (Ypred >= t).astype(int),
                                           termIC,
                                           avg=True,
                                           returnRuMi=False)

    return np.min(ss)


def normalizedSemanticDistance(Ytrue,
                               Ypred,
                               termIC,
                               avg=False,
                               returnRuMi=False):
    """evaluate a set of protein predictions using normalized semantic distance
    value of 0 means perfect predictions, larger values denote worse
    predictions,

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, predicted binary label ndarray (not compressed). Must have hard predictions (0 or 1, not posterior probabilities)
        termIC: output of ic function above

    OUTPUT:
        depending on returnRuMi and avg. To get the average sd over all proteins in a batch/dataset
        use avg = True and returnRuMi = False
        To get result per protein, use avg = False
    """

    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru**2 + mi**2)

    if avg:
        ru = np.mean(ru)
        mi = np.mean(mi)
        sd = np.sqrt(ru**2 + mi**2)

    if not returnRuMi:
        return sd

    return [ru, mi, sd]


def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nru = num / denom

    if avg:
        nru = np.mean(nru)

    return nru


def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nmi = num / denom

    if avg:
        nmi = np.mean(nmi)

    return nmi


def bootstrap(Ytrue,
              Ypred,
              ic,
              nrBootstraps=1000,
              nrThresholds=51,
              seed=1002003445):
    """perform bootstrapping (https://en.wikipedia.org/wiki/Bootstrapping) to
    estimate variance over the test set. The following metrics are used:
    protein-centric average precision, protein centric normalized semantic
    distance, term-centric roc auc.

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        termIC: output of ic function above
        nrBootstraps: the number of bootstraps to perform
        nrThresholds: the number of thresholds to check for calculating smin.

    OUTPUT:
        a dictionary with the metric names as keys (auc, roc, sd) and the bootstrap results as values (nd arrays)
    """

    np.random.seed(seed)
    seedonia = np.random.randint(low=0, high=4294967295, size=nrBootstraps)

    bootstraps_psd = np.zeros((nrBootstraps, ), float)
    bootstraps_pauc = np.zeros((nrBootstraps, ), float)
    bootstraps_troc = np.zeros((nrBootstraps, ), float)
    bootstraps_pfmax = np.zeros((nrBootstraps, ), float)

    for m in range(nrBootstraps):
        [newYtrue, newYpred] = resample(Ytrue, Ypred, random_state=seedonia[m])

        bootstraps_pauc[m] = average_precision_score(newYtrue,
                                                     newYpred,
                                                     average='samples')
        bootstraps_psd[m] = smin(newYtrue, newYpred, ic, nrThresholds)

        tokeep = np.where(np.sum(newYtrue, 0) > 0)[0]
        newYtrue = newYtrue[:, tokeep]
        newYpred = newYpred[:, tokeep]

        tokeep = np.where(np.sum(newYtrue, 0) < newYtrue.shape[0])[0]
        newYtrue = newYtrue[:, tokeep]
        newYpred = newYpred[:, tokeep]

        bootstraps_troc[m] = roc_auc_score(newYtrue, newYpred, average='macro')
        bootstraps_pfmax[m] = fmax(newYtrue, newYpred, nrThresholds)

    return {
        'auc': bootstraps_pauc,
        'sd': bootstraps_psd,
        'roc': bootstraps_troc,
        'fmax': bootstraps_pfmax
    }


def avg_p_r_c(y_true, y_pred, thresholds):
    y_true = y_true[None]
    y_pred = y_pred[None]
    mtx = thresholds[:, None, None] <= (y_pred)

    pred_as_true = mtx.sum(-1)
    real_true = y_true.sum(-1)

    tp = ((y_true == 1) & mtx).sum(-1)
    deno = pred_as_true + 1e-10
    # deno[deno == 0] = 1
    p = tp / deno

    deno = real_true + 1e-10
    # deno[deno == 0] = 1
    r = tp / deno

    n_above_thresholds = np.any(mtx, axis=-1).sum(-1)

    deno = n_above_thresholds + 1e-10
    # deno[deno == 0] = 1
    precisions = p.sum(-1) / deno

    deno = y_true.shape[1]
    recalls = r.sum(-1) / deno

    assert precisions.shape == thresholds.shape
    assert recalls.shape == thresholds.shape

    return precisions, recalls, thresholds


def compute_aupr(true_labels, pred_scores):
    available_index = true_labels.sum(0).astype('bool')
    true_labels = true_labels[:, available_index]
    pred_scores = pred_scores[:, available_index]

    aupr = average_precision_score(true_labels, pred_scores, average='macro')

    return aupr


def do_compute_metrics(true_labels, pred_scores):

    available_index = true_labels.sum(0).astype('bool')
    true_labels = true_labels[:, available_index]
    pred_scores = pred_scores[:, available_index]

    # ******************protein-centric F_max*****************
    precision = dict()
    recall = dict()
    ap = dict()
    thresholds = dict()
    max_thrs = dict()

    prot_available_index = true_labels.sum(1).astype('bool')
    prot_true_labels = true_labels[prot_available_index]
    prot_pred_scores = pred_scores[prot_available_index]
    for i in range(int(prot_true_labels.shape[0])):
        precision[i], recall[i], thresholds[
            i] = metrics.precision_recall_curve(prot_true_labels[i],
                                                prot_pred_scores[i])
        max_thrs[i] = thresholds[i].max()
        ap[i] = metrics.average_precision_score(prot_true_labels[i],
                                                prot_pred_scores[i])
    all_thresholds = np.linspace(0, max(max_thrs.values()), 1000)
    # print('Max value is ', max(max_thrs.values()), end='\t')
    # print('Min value is ', min(max_thrs.values()))

    precision['macro'], recall['macro'], _ = avg_p_r_c(true_labels,
                                                       pred_scores,
                                                       all_thresholds)
    deno = (precision['macro'] + recall['macro']) + 1e-10
    # deno[deno == 0] = 1
    f1_scores = 2 * (precision['macro'] * recall['macro']) / deno
    f1_max = f1_scores.max()

    # ******************term-centric AUPRC*****************
    auprc = compute_aupr(true_labels, pred_scores)

    return f1_max, auprc


def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns
