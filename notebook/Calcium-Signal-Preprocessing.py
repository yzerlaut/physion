# %% [markdown]
# # Preprocessing of Calcium Signals

# %% [markdown]
# ### Methods
#
#
# Raw fluorescence traces were first corrected for neuropil contamination by subtracting the time-varying neuropil signal of each ROI using a fixed scaling coefficient of 0.7 (Dipoppa et al., 2018). Then, to be able to compare activity across cells and mouse lines, fluorescence signals were then normalized. To do this, we used the ΔF/F0 method, calculated using the following formula: (F-F0)/F0, where F is the fluorescence and F0 is the baseline fluorescence. The baseline fluorescence was calculated by tracking the lower 5th percentile over a sliding window of 2min followed by a Gaussian smoothing with a 2min width. ROIs with baseline fluorescence reaching negative values (i.e. with too high neuropil signals to isolate cellular activity) were discarded from the analysis.
#
#
#
# see the function `compute_dFoF` in  [Calcium.py](../src/physion/imaging/Calcium.py)

# %%
import sys, os
import numpy as np
from scipy import stats

sys.path += ['../src'] # add src code directory for physion

import physion
import matplotlib.pylab as plt
import physion.utils.plot_tools as pt
pt.set_style('dark')

# %% [markdown]
# ## Baseline determination to compute ∆F/F (on synthetic data)
#
# I illustrate below a few different options to determine the baseline of a signal

# %%
N = 5 # rois
t = np.linspace(0, 1, int(1e3))
x = np.random.randn(N,len(t))+2*(1-t) 
sWindow = 30
percentile = 10

# %%
from physion.imaging.Calcium import compute_percentile,\
                                    compute_minimum

fig, AX = plt.subplots(1, 2, figsize=(10,2))

for ax, title, x0 in zip(AX,
                  ['fixed %i$^{th}$ percentile' % percentile, 'fixed minimum'],
                  [compute_percentile(x, percentile),
                   compute_minimum(x)]):
    ax.set_title(title)
    for roi in range(N):
        ax.plot(t, 6*roi+x[roi,:], color='tab:green')
        ax.plot(t, 6*roi+x0[roi,:], color='tab:red')

# %%
from physion.imaging.Calcium import compute_sliding_percentile,\
                                    compute_sliding_minimum


fig, AX = plt.subplots(2, 2, figsize=(10,4))

SMOOTHING = False
for ax, title, x0 in zip(AX[0],
                  ['sliding percentile', 'sliding minimum'],
                  [compute_sliding_percentile(x, percentile, sWindow, with_smoothing=SMOOTHING),
                   compute_sliding_minimum(x, sWindow, with_smoothing=SMOOTHING)]):
    ax.set_title(title)
    for roi in range(N):
        ax.plot(t, 6*roi+x[roi,:], color='tab:green')
        ax.plot(t, 6*roi+x0[roi,:], color='tab:red')
        
SMOOTHING = True
for ax, x0 in zip(AX[1],
                  [compute_sliding_percentile(x, percentile, sWindow, with_smoothing=SMOOTHING),
                   compute_sliding_minimum(x, sWindow, with_smoothing=SMOOTHING)]):
    for roi in range(N):
        ax.plot(t, 6*roi+x[roi,:], color='tab:green')
        ax.plot(t, 6*roi+x0[roi,:], color='tab:red')
AX[0][0].set_ylabel('no smoothing')
AX[1][0].set_ylabel('with smoothing')

# %% [markdown]
# ## Illustration of discarding criteria

# %%
from physion.imaging.Calcium import compute_F0

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2023_02_15-13-30-47.nwb')
data = physion.analysis.read_NWB.Data(filename)

# %%
dFoF_options = dict(\
    method_for_F0='percentile',
    percentile=5.,
    roi_to_neuropil_fluo_inclusion_factor=1.15,
    neuropil_correction_factor=0.8,
    with_computed_neuropil_fact=False,
    roi_to_neuropil_fluo_inclusion_factor_metric='mean')

# %%
# we first perform the dFoF determination with the above params
#    (this restrict the available ROIs in the future)
data.build_dFoF(**dFoF_options, verbose=True)
valid = data.valid_roiIndices
rejected = [i for i in range(data.Fluorescence.data.shape[1]) if (i not in valid)]
# we re-initialize the fluo and neuropil to get back to all ROIs
data.initialize_ROIs(valid_roiIndices=None)
data.build_rawFluo()
data.build_neuropil()

# %%
for roi in np.concatenate([np.random.choice(valid, 7, replace=False),
                           np.random.choice(rejected, min([3, len(rejected)]), replace=False)]):
    fig, ax = plt.subplots(figsize=(9,1.))
    plt.plot(data.t_dFoF, data.neuropil[roi,:], label='Neuropil', color='tab:orange')
    plt.plot(data.t_dFoF, data.rawFluo[roi,:], label='ROI Fluo.', color='tab:green')
    plt.title('ROI#%i' % (roi+1)+('--> valid' if roi in valid else '--> rejected'))
    plt.legend()
    plt.ylabel('Fluo. (a.u.)')
plt.xlabel('time (s)');

# %%
correctedFluo = data.rawFluo-data.neuropil_correction_factor*data.neuropil
baseline = physion.imaging.Calcium.compute_F0(data, correctedFluo, 
                                              method=dFoF_options['method_for_F0'],
                                              percentile=dFoF_options['percentile'])
np.random.seed(1)
for roi in np.concatenate([np.random.choice(valid, 7, replace=False),
                           np.random.choice(rejected, min([3, len(rejected)]), replace=False)]):
    fig, AX = plt.subplots(1, 2, figsize=(9,1.))
    AX[0].plot(data.t_dFoF, correctedFluo[roi,:], label='ROI Fluo.', color='tab:blue')
    AX[0].plot(data.t_dFoF, baseline[roi,:], label='F0', color='tab:red')
    dFoF = (correctedFluo[roi,:]-baseline[roi,:])/baseline[roi,:]
    try:
        AX[1].plot(data.t_dFoF, dFoF, color='tab:green')
    except BaseException as be:
        pass
    fig.suptitle('ROI#%i' % (roi+1)+('--> valid' if roi in valid else '--> rejected'))
    AX[0].legend(frameon=False, fontsize='x-small')
    AX[0].set_ylabel('corr. Fluo. (a.u.)')
plt.xlabel('time (s)');

# %% [markdown]
# ## 
# Deconvolution

# %%
from physion.imaging.dcnv import oasis, preprocess

TAU = 1.3 # s , decay of the Calcium Indicator, here GCamp6s

dFoF_options = dict(\
    method_for_F0='sliding_percentile',
    percentile=10.,
    sliding_window=120.0,
    roi_to_neuropil_fluo_inclusion_factor=1.15,
    neuropil_correction_factor=0.8)

data.build_dFoF(**dFoF_options)

Dcnv = oasis(data.dFoF, len(data.t_dFoF), TAU, 1./data.CaImaging_dt)

# %%
tzoom = [10,50]

np.random.seed(1)
for roi in np.random.choice(range(data.nROIs), 10, 
                            replace=False):
    fig, ax = plt.subplots(1, figsize=(10,1))
    ax2 = ax.twinx()
    cond = (data.t_dFoF>tzoom[0]) & (data.t_dFoF<tzoom[1])
    ax.plot(data.t_dFoF[cond], data.dFoF[roi,cond], label='$\Delta$F/F', color='tab:green')
    ax2.plot(data.t_dFoF[cond], Dcnv[roi,cond], label='Deconv.', color='tab:blue')
    fig.suptitle('ROI#%i' % (roi+1))
    ax.annotate('$\Delta$F/F', (1,1), va='top', xycoords='axes fraction', color='tab:green')
    ax2.annotate('\nDeconvolved', (1,1), va='top', xycoords='axes fraction', color='tab:blue')
    ax.set_ylabel('(a.u.)')
    ax.axis('off')
    ax2.axis('off')
    ax.plot(tzoom[0]+np.arange(2), ax.get_ylim()[1]*np.ones(2), 'k-')
    ax.annotate('1s', (tzoom[0], ax.get_ylim()[1]))
    
ax.set_xlabel('time (s)')

# %% [markdown]
# ## Fast sliding percentile

# %%
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import filters
from scipy.interpolate import interp1d

nROIs, N = 5, 100000
x = np.random.randn(nROIs,N)
x[1,:] = np.linspace(-3,3,N) +np.random.randn(N)
x[3,:] = np.linspace(3,-3,N) +np.random.randn(N)

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    """ 
    for sliding window analysis, see: 
    https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    """
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a,
                        shape=(nrows,L), strides=(S*n,n))

def sliding_percentile(array, percentile, Window):

    x = np.zeros(len(array))

    # using a sliding "view" of the array
    y0 = strided_app(array, Window, 1)
    
    y = np.percentile(y0, percentile, axis=-1)
    
    # clean up boundaries
    x[:int(Window/2)] = y[0]
    x[int(Window/2):int(Window/2)+len(y)] = y
    x[-int(Window/2):] = y[-1]

    return x
    
def compute_sliding_percentile(array, percentile, Window,
                               subsampling_window_factor=0.1,
                               with_smoothing=True):
    """
    sliding percentile over a window
            with subsampling to make it more efficient
            subsampling_window_factor=0 -> no subsampling !
    """

    subsampling = max([1,int(subsampling_window_factor*Window)])
    print(subsampling)
    Flow = np.zeros(array.shape)
    indices = np.arange(array.shape[1])
    sbsmplIndices = (indices%subsampling)==0
    for roi in range(array.shape[0]):
        Flow[roi,sbsmplIndices] = sliding_percentile(array[roi,sbsmplIndices], percentile,
                                                     max([1,int(Window/subsampling)]))

    if with_smoothing:
        Flow[:,sbsmplIndices] = filters.gaussian_filter1d(Flow[:,sbsmplIndices], 
                                                          max([1,int(Window/subsampling)]), 
                                                          axis=-1)

    Flow[:,~sbsmplIndices] = interp1d(indices[sbsmplIndices], Flow[:,sbsmplIndices],
                                      kind='linear', fill_value='extrapolate',
                                      axis=-1)(indices[~sbsmplIndices])

    return Flow

import time

tic = time.time()
Flow = compute_sliding_percentile(x, 5, 4000,
                                  subsampling_window_factor=1e-1,
                                  with_smoothing=True)
print('time: %.2fs' % (time.time()-tic))

fig, AX = plt.subplots(nROIs, 1, figsize=(8,5))
for i in range(nROIs):
    AX[i].plot(x[i, ::5])
    AX[i].plot(Flow[i, ::5])
    AX[i].axis('off')

# %%
