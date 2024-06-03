from .fast import enm_proba_exact
from .fast import enm_proba_apprx

import numpy as np
from sklearn.svm import SVC

import tqdm

LEFT = 1e-3
RIGHT = 1e+3
EPS = 1e-4


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
    prb = 0
    iters = int(1 / eps**2)
    for i in tqdm.tqdm(range(iters), total=iters):
        score = svm_classify(
                X,
                y[np.random.permutation(len(y))]
        )

        if score <= k:
            prb += 1

    return prb, iters
