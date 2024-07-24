import numpy as np
import pandas as pd
import tqdm
import time
import pickle

from scipy.special import comb as binom
from lib import svm_proba
from lib import enm_proba_exact
from lib import enm_proba_apprx

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 8098
np.random.seed(SEED)

N_THREADS = 2

ITERATIONS = 10
N_MIN = 1
N_MAX = 31
K = 1

N_PROCESSES = 32

def timer(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

def run(N0, N1, k):
    X0 = np.random.multivariate_normal(
        [0, 0],
        [[1, 0], [0, 1]],
        size=N0
    )
    y0 = np.zeros(N0, dtype=int) + 0

    X1 = np.random.multivariate_normal(
        [0, 0],
        [[1, 0], [0, 1]],
        size=N1
    )
    y1 = np.zeros(N1, dtype=int) + 1

    X = np.vstack([X0, X1], dtype=np.float32)
    y = np.concatenate([y0, y1], dtype=np.int32)

    result, time = timer(enm_proba_exact, X, y, k, N_THREADS)
    nominator, denominator = result

    return N0, N1, nominator, denominator, time


for k in range(K):
    futures = []
    bank = []
    
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        for lhs in range(N_MIN, N_MAX):
            for rhs in range(lhs, N_MAX):
                for _ in range(ITERATIONS):
                    futures.append(pool.submit(run, lhs, rhs, k))
        
        for future in tqdm.tqdm(futures):
            bank.append(future.result())

            df = pd.DataFrame(bank, columns=["lhs", "rhs", "nominator", "denominator", "time"])
            df.to_csv(f"data/normal_{N_MIN}_{N_MAX}_{k}.tsv", sep="\t", index=None)

    print(k)
