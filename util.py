import numpy as np
from scipy.misc import comb

def rand_score(clusters, classes):

    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def findMost(I, n):
    v = []
    for i1 in range(len(I)):
        v.append(np.sort(np.concatenate([I[i1][0:i1],I[i1][i1+1:]]))[n-1])
    return np.array(v)
