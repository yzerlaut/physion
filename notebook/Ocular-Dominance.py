# %% [markdown]
# # Ocular Dominance analysis

# %%
# load packages:
import numpy as np
import sys
sys.path += ['../src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from physion.intrinsic.tools import *
import matplotlib.pylab as plt
from PIL import Image

def preprocess(datafolder,
               side='left',
               temporal_smoothing=1,
               spatial_smoothing=8,
               F0_percentile=5.):

    output = {}
    for key in ['up', 'down']:

        params, (t, data) = load_raw_data(datafolder, 
                                        '%s-%s' % (\
                                            side, key))

        # spatial and temporal smoothing
        pData = resample_img(data, spatial_smoothing) # pre-processed data
        if temporal_smoothing>0:
                pData = gaussian_filter1d(pData, temporal_smoothing, axis=0)
        
        # Fourier analysis
        spectrum = np.fft.fft(pData, axis=0)
        # power and phase at the stimulus frequency (i.e. # repeats)
        power = np.abs(spectrum)[params['Nrepeat'], :, :]/data.shape[0]
        
        output['meanFluo_%s' % key] = pData.mean(axis=0)
        output['stdFluo_%s' % key] = pData.std(axis=0)
        output['powerFluo_%s' % key] = power

        pData += 10 # to insure a decent baseline

        # delta F / F computation
        F0 = np.percentile(pData, F0_percentile, axis=0)
        dFoF = (pData-F0)/F0
        # Fourier analysis of dFoF
        spectrum = np.fft.fft(dFoF, axis=0)
        power = np.abs(spectrum)[params['Nrepeat'], :, :]/data.shape[0]

        output['meanDFoF_%s' % key] = dFoF.mean(axis=0)
        output['stdDFoF_%s' % key] = dFoF.std(axis=0)
        output['powerDFoF_%s' % key] = power

    return output

blind = preprocess(\
        #        os.path.expanduser('~/DATA/2025_12_18/14-06-53'))
               os.path.expanduser('~/DATA/2025_12_18/14-31-14'))

ipsi2 = preprocess(\
               os.path.expanduser('~/DATA/2025_12_18/15-03-55'))

contra2 = preprocess(\
               os.path.expanduser('~/DATA/2025_12_18/14-52-05'))


# %%
fig, AX = pt.figure(axes=(3, 6), right=5., 
                    hspace=0.6, wspace=0.1)

bounds = [(0,3000), 
          (0,200), 
          (0,40),
          (0.,0.15),
          (0, .1),
          (0, .03)]

for d, data, title in zip(range(3),
     [ipsi, ipsi2, contra2],
     ['ipsi-eye', 'ipsi-2', 'contra-2']):

        for k, key in enumerate(['Fluo', 'DFoF']):

                for i, prop in enumerate(\
                      ['mean', 'std', 'power']):

                        im = AX[3*k+i][d].imshow(\
                                .5*(data['%s%s_up' % (prop, key)]+\
                                data['%s%s_down' % (prop, key)]),
                                        cmap=plt.cm.binary,
                                        vmin=bounds[3*k+i][0],
                                        vmax=bounds[3*k+i][1])

                        if d==2:
                                fig.colorbar(im, ax=AX[3*k+i][2],
                                                shrink=0.8, aspect=10,
                                                label='Fluo. (a.u.)' if k==0 else '$\Delta$F/F')

        pt.annotate(AX[0][d], title, (0.5,1), ha='center')

for i, label in enumerate(\
      ['mean raw Fluo.', 'std raw Fluo.', 'raw Fluo. power\n@ stim. freq.']+\
      ['mean $\Delta$F/F', 'std $\Delta$F/F', '$\Delta$F/F power\n@ stim. freq.']):
      pt.annotate(AX[i][0], label, (-0.1,0.5), 
                  ha='right', va='center')

for ax in pt.flatten(AX):
      ax.axis('off')


# %%

def plot_power_map(ax, fig, Map,
                   bounds=[0.,0.02]):

    im = ax.imshow(Map, cmap=plt.cm.binary,
                   vmin=bounds[0], vmax=bounds[1])
    fig.colorbar(im, ax=ax,
                 shrink=0.4,
                 aspect=10,
                 label='$\Delta$F/F power ')

def make_fig(IMAGES):


    fig, AX = plt.subplots(3, 2, figsize=(7,5))
    plt.subplots_adjust(wspace=0.8, right=0.8, bottom=0.1)

    plot_power_map(AX[0][0], fig, IMAGES['ipsi-power'])
    AX[0][0].set_title('Ipsi power')
    plot_power_map(AX[0][1], fig, IMAGES['contra-power'])
    AX[0][1].set_title('Contra power')
    for ax in AX[0]:
        ax.axis('off')

    plot_power_map(AX[1][0], fig, IMAGES['ipsi-power-thresh'])
    AX[1][0].set_title('thresh. Ipsi ')
    plot_power_map(AX[1][1], fig, IMAGES['contra-power-thresh'])
    AX[1][1].set_title('thresh. Contra')
    for ax in AX[1]:
        ax.axis('off')

    im = AX[2][0].imshow(IMAGES['ocular-dominance'],
                        cmap=plt.cm.twilight, vmin=-0.5, vmax=0.5)
    cbar = fig.colorbar(im, ax=AX[2][0],
                        ticks=[-0.5, 0, 0.5], 
                        shrink=0.4, aspect=10, label='OD index')
    AX[2][0].axis('off')
    AX[2][0].set_title('Ocular Dominance')

    AX[2][1].hist(IMAGES['ocular-dominance'].flatten(),
                bins=np.linspace(-1, 1, 150))
    AX[2][1].set_xlabel('OD index')
    AX[2][1].set_ylabel('pix. count')
    AX[2][1].set_title('mean OD index: %.2f' % \
            np.nanmean(IMAGES['ocular-dominance']))
    
    return fig, AX


threshOD = 0.5


# ----------------------------------- #
#               power maps            #
# ----------------------------------- #
maps = {}
maps['ipsi-power'] = 0.5*(\
        ipsi['powerDFoF_up']+\
        ipsi['powerDFoF_down'])

maps['contra-power'] = 0.5*(\
        contra['powerDFoF_up']+\
        contra['powerDFoF_down'])

# ----------------------------------- #
#           threshold power           #
# ----------------------------------- #

thresh = threshOD*np.max(maps['ipsi-power'])
threshCond = maps['ipsi-power']>thresh

maps['ipsi-power-thresh'] = -np.ones(\
        maps['ipsi-power'].shape)*np.nan
maps['ipsi-power-thresh'][threshCond] = \
        maps['ipsi-power'][threshCond]
maps['contra-power-thresh'] = -np.ones(\
        maps['contra-power'].shape)*np.nan
maps['contra-power-thresh'][threshCond] = \
        maps['contra-power'][threshCond]


# ----------------------------------- #
#           ocular dominance          #
# ----------------------------------- #
maps['ocular-dominance'] = -np.ones(\
        maps['contra-power'].shape)*np.nan
maps['ocular-dominance'][threshCond] = \
        (maps['contra-power'][threshCond]-\
                maps['ipsi-power'][threshCond])/\
        (maps['contra-power'][threshCond]+\
                maps['ipsi-power'][threshCond])

fig, AX = make_fig(maps)
print(' --> ok')


        
# %%
