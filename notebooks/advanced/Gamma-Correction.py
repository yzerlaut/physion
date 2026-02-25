# %%
import sys, time
sys.path += ['../../src']
import numpy as np
import pandas as pd

import physion.utils.plot_tools as pt
pt.set_style('dark')
from scipy.optimize import minimize 

# %%
data = pd.read_csv(os.path.join(\
    '..', 'data', 'Gamma-Correction-Measurements.csv'))

# %%

x = np.linspace(0, 1, len(data['Value']))

fig, ax = pt.figure()
ax.plot(x, data['Value'], 'o-', ms=2, lw=0.2)
pt.set_plot(ax, xticks=[0., 0.5, 1.],
            xlabel='screen color', ylabel='Candela')

# %%

normed_data = \
    (data['Value']-data['Value'].min())/(data['Value'].max()-data['Value'].min())

def func(lum, coefs):
    return coefs[0]*lum**coefs[1]

def to_minimize(coefs):
    return np.sum(np.abs(normed_data-func(x, coefs))**2)

residual = minimize(to_minimize, [1, 1],
                    bounds=[(0.5, 2), (0.1, 3.)])

fig, ax = pt.figure()
ax.scatter(x, normed_data, label='data', s=2)
pt.set_plot(ax,
    title="k=%.2f, $\gamma$=%.2f" % (residual.x[0], residual.x[1]),
            )
ax.plot(x, func(x, residual.x), lw=3, alpha=.5, label='fit')
ax.legend(frameon=False, loc=(1.0, 0.2))
pt.set_plot(ax, 
            xticks=[0., 0.5, 1.], yticks=[0., 0.5, 1.],
            xlabel='screen color', 
            ylabel='normed\nluminance')

# %%
fig, ax = pt.figure()
ax.plot(x, data['After'], 'o-', ms=2)
pt.set_plot(ax, xticks=[0., 0.5, 1.],
            xlabel='screen color', ylabel='Candela',
            title='After Gamma Correction')

# %%
fig, AX = pt.figure(axes=(3,1), wspace=2)
AX[0].plot(x, data['Value'], 'o-', ms=2, lw=0.2)
pt.set_plot(AX[0], xticks=[0., 0.5, 1.],
            xlabel='screen color', ylabel='Candela',
            title='Before Correction')
AX[1].scatter(x, normed_data, label='data', s=2)
pt.set_plot(AX[1],
    title="fit: k=%.2f, $\gamma$=%.2f" % (residual.x[0], residual.x[1]),
            )
AX[1].plot(x, func(x, residual.x), lw=3, alpha=.5, label='fit')
AX[1].legend(frameon=False, loc=(0, 0.5))
pt.set_plot(AX[1], 
            xticks=[0., 0.5, 1.], yticks=[0., 0.5, 1.],
            xlabel='screen color', 
            ylabel='normed\nluminance')

AX[2].plot(x, data['After'], 'o-', ms=2)
pt.set_plot(AX[2], xticks=[0., 0.5, 1.],
            xlabel='screen color', ylabel='Candela',
            title='After Gamma Correction')
fig.savefig('../../docs/visual_stim/gamma-correction.png')
# %%

