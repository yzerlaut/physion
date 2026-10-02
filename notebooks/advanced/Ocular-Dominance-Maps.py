# %% [markdown]
# # Ocular Dominance maps
# 
# from:
# 
# > Cang, J., Kalatsky, V. A., Löwel, S., & Stryker, M. P. (2005). 
# >
# > Optical imaging of the intrinsic signal as a measure of cortical plasticity in the mouse. 
# >
# > Visual neuroscience, 22(5), 685-691.

# %%

import sys
import numpy as np
sys.path += ['../../src']
from physion.intrinsic.ocular_dominance import plot_power_map
import physion.utils.plot_tools as pt

maps = np.load(os.path.join(os.path.expanduser('~'), 'DATA',
                            'OD-exps', 'contra-eye', 
                            'ocular-dominance-maps.npy'),
                            allow_pickle=True).item()
contra_eye_power = \
    .5*(maps['left-up-power']+maps['left-down-power'])
maps = np.load(os.path.join(os.path.expanduser('~'), 'DATA',
                            'OD-exps', 'ipsi-eye', 
                            'ocular-dominance-maps.npy'),
                            allow_pickle=True).item()
ipsi_eye_power = \
    .5*(maps['left-up-power']+maps['left-down-power'])

# threshold artefact
for array in [contra_eye_power, ipsi_eye_power]:
    array[array>3e-4] = 0

# print(np.max(contra_eye_power))

# %%
# filtering
from scipy.ndimage import gaussian_filter
contra_eye_power = gaussian_filter(contra_eye_power, 2)
ipsi_eye_power = gaussian_filter(ipsi_eye_power, 2)


# %%
import matplotlib.pylab as plt
fig, AX = pt.figure(axes=(2,1), ax_scale=(1.5,2.))
plot_power_map(AX[0], fig, contra_eye_power,
               bounds=[0,4])
AX[0].set_title('contra eye')
plot_power_map(AX[1], fig, ipsi_eye_power,
               bounds=[0,4])
AX[1].set_title('ipsi eye')
for ax in AX:
    ax.axis('off')
# %%

thresh = 0.5*np.max(ipsi_eye_power)
threshCond = ipsi_eye_power>thresh

ocular_dominance = -np.ones(\
    ipsi_eye_power.shape)*np.nan
ocular_dominance[threshCond] = \
    (contra_eye_power[threshCond]-\
        ipsi_eye_power[threshCond])/\
    (contra_eye_power[threshCond]+\
        ipsi_eye_power[threshCond])
fig, ax = pt.figure(ax_scale=(1.5,2.2), right=8)
inset = pt.inset(ax, [1.7,0.2,0.5,.6])
im = ax.imshow(ocular_dominance,
          cmap=plt.cm.twilight, vmin=-1, vmax=1)
cbar = fig.colorbar(im, ax=ax,
            ticks=[-1, 0, 1], 
            shrink=0.4,
            aspect=15,
            label='OD value')
# cbar.ax.set_yticklabels(['-$\\pi$', '0', '$\\pi$'])
ax.axis('off')
inset.hist(ocular_dominance.flatten(),
           color='grey')
vAbs = np.nanmax(np.abs(ocular_dominance))
pt.set_plot(inset, 
            xlim=[-vAbs,vAbs],
            xticks=[float('%.1f'%f) for f in [-1.5*vAbs,0,1.5*vAbs]],
            xlabel='OD value',
            ylabel='pix. count',
            title='mean: %.2f' % np.nanmean(ocular_dominance))
# %%
