#Bootstrapping Significant Test

import numpy as np, pandas as pd
from pylab import *
import math
np.random.seed(1234321)

#How many resampling?
times = 500

def bootstrap_resample(X, n):
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

#X1 is the samples from group 1
X1 = [
90.98,
90.87,
90.88,
91.04,
90.82,
90.75,
90.92,
91.1,
90.87,
90.92,
90.94,
90.73,
90.89
]

#X2 is the samples from group 2
X2 = [
90.88,
90.79,
90.21,
90.56,
90.62,
90.74,
90.86,
91.16,
90.58,
90.98,
90.6,
90.89,
90.77
]

n = float(len(X1))
X1 = np.array(X1)
varX1 = np.var(X1, dtype=np.float64)
meanX1 = np.mean(X1)

m = float(len(X2))
X2 = np.array(X2)
varX2 = np.var(X2, dtype=np.float64)
meanX2 = np.mean(X2)

SD = np.sqrt(varX1/n + varX2/m)
T = (meanX1 - meanX2)/SD
print T

TT = np.zeros((times,))
for i in range(times):
    X1_resample = bootstrap_resample(X1, int(n))
    varX1_resample = np.var(X1_resample, dtype=np.float64)
    meanX1_resample = np.mean(X1_resample)

    X2_resample = bootstrap_resample(X2, int(m))
    varX2_resample = np.var(X2_resample, dtype=np.float64)
    meanX2_resample = np.mean(X2_resample)

    SDD = np.sqrt(varX1_resample/n + varX2_resample/m)
    tt = ((meanX1_resample - meanX2_resample) - (meanX1 - meanX2))/SDD
    TT[i] = tt

sortedTT = np.sort(TT)
Bin = list(np.arange(sortedTT.min()-10, sortedTT.max()+10, 0.01))
F = np.zeros((len(Bin),))
for i in range(len(Bin)):
    s = 0.0
    for j in TT:
        if j <= Bin[i]:
            s += 1.0
    F[i] = s/len(TT)

for i in range(len(Bin)):
    if F[i]>=0.95:
        t1=Bin[i]
        break

for i in range(len(Bin)):
    if F[i]>=0.90:
        t2=Bin[i]
        break
for i in range(len(Bin)):
    if Bin[i]>=T:
        p_value = 1 - F[i]
        break

print 't-score for 0.05: ', t1
print 't-score for 0.1: ', t2
print 'p-value: ', p_value
