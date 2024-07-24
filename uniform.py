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

N_THREADS = 4

ITERATIONS = 1
N_MIN = 1
N_MAX = 31
K = 6

N_PROCESSES = 16

def timer(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

def run(N0, N1, k):
    X0 = np.random.uniform(
        size=(N0, 2)
    )
    y0 = np.zeros(N0, dtype=int) + 0

    X1 = np.random.uniform(
        size=(N1, 2)
    )
    y1 = np.zeros(N1, dtype=int) + 1

    X = np.vstack([X0, X1], dtype=np.float32)
    y = np.concatenate([y0, y1], dtype=np.int32)

    result, time = timer(enm_proba_exact, X, y, k, N_THREADS)
    nominator, denominator = result

    # return N0, N1, nominator, denominator, time
    return N0, nominator, denominator, time


for k in range(K):
    futures = []
    bank = []
    
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        for lhs in list(range(N_MIN, N_MAX))[::-1]:
            # for rhs in list(range(lhs, N_MAX))[::-1]:
            #     for _ in range(ITERATIONS):
            #         futures.append(pool.submit(run, lhs, rhs, k))
            futures.append(pool.submit(run, lhs, lhs, k))
        
        for future in tqdm.tqdm(futures):
            bank.append(future.result())

            # df = pd.DataFrame(bank, columns=["lhs", "rhs", "nominator", "denominator", "time"])
            df = pd.DataFrame(bank, columns=["size", "nominator", "denominator", "time"])
            df.to_csv(f"data/uniform_{N_MIN}_{N_MAX}_{k}.tsv", sep="\t", index=None)

    print(k)
