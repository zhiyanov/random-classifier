from .fast import enm_proba_exact as enm_proba_exact_
from .fast import enm_proba_apprx

import numpy as np
from sklearn.svm import SVC
from scipy.special import comb

import tqdm

LEFT = 1e-3
RIGHT = 1e+3
EPS = 1e-4

SEED = 42
np.random.seed(SEED)

def binary(func, left, right, eps):
    while right - left > eps:
        mid = (left + right) / 2
        value = func(mid)

        if func(mid) > 0:
            right = mid
        elif func(mid) < 0:
            left = mid
        else:
            return mid
    
    return mid

def svm_classify(X, y):
    svm = SVC(C=RIGHT,
              kernel="linear")
    svm.fit(X, y)

    return len(X) * (1 - svm.score(X, y))

def svm_proba(X, y, k, eps):
    prb_less, prb_equal = 0, 0
    iters = int(1 / eps**2)
    for i in tqdm.tqdm(range(iters), total=iters):
        score = svm_classify(
                X,
                y[np.random.permutation(len(y))]
        )

        if score <= k:
            prb_equal += 1

        if score < k:
            prb_less += 1

    return prb_equal - prb_less, iters

def enm_proba_exact(X, y, k, parallel):
    n = y.sum()
    m = len(y) - y.sum()
    nominator, _ = enm_proba_exact_(X, y, k, parallel)
    return nominator, comb(n + m, n, exact=True)
