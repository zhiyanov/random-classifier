from .fast import enm_proba_exact as enm_proba_exact_
from .fast import enm_proba_apprx

import numpy as np
from sklearn.svm import SVC
from scipy.special import comb

from concurrent.futures import ProcessPoolExecutor

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

def svm_exect(X, y, k, iters, visual):
    if visual:
        iterator = tqdm.tqdm(range(iters), total=iters)
    else:
        iterator = range(iters)
    
    prb_equal, prb_less = 0, 0
    for i in iterator:
        score = svm_classify(
            X,
            y[np.random.permutation(len(y))]
        )

        if score <= k:
            prb_equal += 1

        if score < k:
            prb_less += 1

    return prb_equal, iters

def svm_proba(X, y, k, eps, parallel):
    prb_less, prb_equal = 0, 0
    iters = int(1 / eps**2 / parallel)
    
    futures = []
    bank = []

    with ProcessPoolExecutor(parallel) as pool:
        for proc in range(parallel):
            if proc:
                futures.append(pool.submit(
                    svm_exect, X, y, k, iters, False))
            else:
                futures.append(pool.submit(
                    svm_exect, X, y, k, iters, True))

        for future in futures:
            bank.append(future.result())
    
    nominator, denominator = 0, 0
    for nom, denom in bank:
        nominator += nom
        denominator += denom

    return nominator, denominator

def enm_proba_exact(X, y, k, parallel):
    n = y.sum()
    m = len(y) - y.sum()
    nominator, _ = enm_proba_exact_(X, y, k, parallel)
    return nominator, comb(n + m, n, exact=True)
