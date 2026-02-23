# %%
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize 

# %%
data = pd.read_csv(os.path.expanduser(\
    '~/Desktop/Visual-Stim/Gamma-Correction-Measurements.csv'))

# %%
x = np.linspace(0, 1, len(data['Value']))
plt.plot(x, data['Value'], 'o-')
plt.xlabel('screen color')
plt.ylabel('Candela')

# %%

normed_data = \
    (data['Value']-data['Value'].min())/(data['Value'].max()-data['Value'].min())

def func(lum, coefs):
    return coefs[0]*lum**coefs[1]

def to_minimize(coefs):
    return np.sum(np.abs(normed_data-func(x, coefs))**2)

residual = minimize(to_minimize, [1, 1],
                    bounds=[(0.5, 2), (0.1, 3.)])

plt.title("k=%.2f, $\gamma$=%.2f" % (residual.x[0], residual.x[1]))
plt.scatter(x, normed_data, label='data', s=3)
plt.plot(x, func(x, residual.x), lw=3, alpha=.5, label='fit')
plt.legend()

# %%
x = np.linspace(0, 1, len(data['Value']))
plt.plot(x, data['After'], 'o-')
plt.xlabel('screen color')
plt.ylabel('Candela')

# %%
