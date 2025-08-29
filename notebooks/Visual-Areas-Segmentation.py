# %% [markdown]
# # Visual Area Segmentation
#
# ##### copied/adapted from:
#
# https://github.com/zhuangjun1981/NeuroAnalysisTools/blob/master/NeuroAnalysisTools/RetinotopicMapping.py
#
# ##### Cite the original work/implementation:
#
# [Zhuang et al., Elife (2017)](https://elifesciences.org/articles/18372)

# %%
# load packages:
import numpy as np
import sys
sys.path.append('../src')
import physion.utils.plot_tools as pt
from physion.intrinsic.tools import *
from physion.intrinsic.analysis import RetinotopicMapping
import matplotlib.pylab as plt
from PIL import Image

# %% [markdown]
# ## Load data

# %%
dataFolder = os.path.join(os.path.expanduser('~'), 'DATA', 
                        'physion_Demo-Datasets', 'PV-WT', 'retinotopic_mapping',
                        'PVTOM_BB_5')
# retinotopic mapping data
# maps = np.load(os.path.join(dataFolder, 'raw-maps.npy') , 
#                allow_pickle=True).item()
maps = dict(np.load(os.path.join(dataFolder, 'raw-maps.npz') , 
               allow_pickle=True))
# vasculature picture
imVasc = np.array(Image.open(os.path.join(dataFolder, 'vasculature.tif')))
fig, ax = pt.figure(ax_scale=(2,2))
ax.imshow(imVasc**1, cmap=plt.cm.gray) 
plt.axis('off');

# %% [markdown]
# # Retinotopic Maps

# %%
plot_retinotopic_maps(maps, map_type='altitude');

# %%
plot_retinotopic_maps(maps, map_type='azimuth');

# %% [markdown]
# # Perform Segmentation

# %%
data = build_trial_data(maps)
data['vasculatureMap'] = imVasc[::int(imVasc.shape[0]/data['aziPosMap'].shape[0]),\
                                ::int(imVasc.shape[1]/data['aziPosMap'].shape[1])]
segmentation_params={'phaseMapFilterSigma': 3.,
                     'signMapFilterSigma': 2.,
                     'signMapThr': 0.8,
                     'eccMapFilterSigma': 10.,
                     'splitLocalMinCutStep': 5.,
                     'mergeOverlapThr': 0.4,
                     'closeIter': 3,
                     'openIter': 3,
                     'dilationIter': 15,
                     'borderWidth': 1,
                     'smallPatchThr': 100,
                     'visualSpacePixelSize': 0.5,
                     'visualSpaceCloseIter': 15,
                     'splitOverlapThr': 1.1}
data['params'] = segmentation_params
trial = RetinotopicMapping.RetinotopicMappingTrial(**data)
trial.processTrial(isPlot=False) # TURN TO TRUE TO VISUALIZE THE SEGMENTATION STEPS
_ = trial._getSignMap(onlySMplot=True)

# %%
fig, ax = pt.figure(ax_scale=(5,5))
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.grey, vmin=imVasc.min(), vmax=imVasc.max(), extent=[*ax.get_xlim(), *ax.get_ylim()])
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.axis('off');
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# # Summary Plot

# %%
fig, AX = pt.figure(axes=(5,1), 
                    ax_scale=(.9*2,.9*3), 
                    wspace=0.4)

for map_type, ax, bounds in zip(['altitude', 'azimuth'], AX[:2],
                                [[-50, 50],[-60, 60]]):
    im = ax.imshow(maps['%s-retinotopy' % map_type], cmap=plt.cm.PRGn,\
                   vmin=bounds[0], vmax=bounds[1])
    fig.colorbar(im, ax=ax,
                 label='visual e (deg.)')
    ax.set_title('%s map' % map_type)

ax = AX[2]
im = ax.imshow(trial.signMapf, cmap='jet', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label='phase gradient sign')
ax.set_title('sign map')

ax=AX[-2]
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.grey, vmin=imVasc.min(), vmax=imVasc.max(), 
          extent=[*ax.get_xlim(), *ax.get_ylim()])
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.set_title('w/ vasculature')

ax = AX[-1]
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.grey, 
          vmin=imVasc.min(), vmax=imVasc.max(), 
          extent=[*ax.get_xlim(), *ax.get_ylim()])

for ax in AX:
    ax.axis('equal')
    ax.axis('off')    

fig.savefig(os.path.join(os.path.expanduser('~'), 
                         'Desktop', 'fig.svg'))

# %%

m = 0*maps['altitude-retinotopy']
cond = ( np.abs(maps['altitude-retinotopy'])<7) & ( np.abs(maps['azimuth-retinotopy'])<7)
cond = np.abs(maps['altitude-retinotopy'])<7
cond = np.abs(maps['azimuth-retinotopy'])<7

m[cond] = 1
plt.imshow(m)

# %%
