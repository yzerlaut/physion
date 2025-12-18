# %%

import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path += ['../src'] # add src code directory for physion

import physion.utils.plot_tools as pt
pt.set_style('dark')



# %%

def load_Camera_data(imgfolder, t0=0):

    
    file_list = [f for f in os.listdir(imgfolder) if f.endswith('.npy')]
    _times = np.array([float(f.replace('.npy', '')) for f in file_list])
    _isorted = np.argsort(_times)
    times = _times[_isorted]-t0
    FILES = np.array(file_list)[_isorted]
    nframes = len(times)
    Lx, Ly = np.load(os.path.join(imgfolder, FILES[0])).shape
    if True:
        print('Sampling frequency: %.1f Hz  (datafile: %s)' % (1./np.diff(times).mean(), imgfolder))
        
       
    return times, FILES, nframes, Lx, Ly


smoothing = 2
from scipy.ndimage import gaussian_filter

def compute_resp(datafolder):

    # load visual stim
    visual_stim = np.load(os.path.join(datafolder, 'visual-stim.npy'),
                          allow_pickle=True).item()
    
    NIdaq_tStart = np.load(os.path.join(datafolder, 'NIdaq.start.npy'),
                          allow_pickle=True).item()
    
    # load
    times, FILES, nframes, Lx, Ly = load_Camera_data(\
                                os.path.join(datafolder, 'ImagingCamera-imgs'),
                                t0=NIdaq_tStart)
    
    dt = 0.1 # seconds
    tstart, tstop = -2, 5

    nt = int((tstop-tstart)/dt)
    t = tstart+np.arange(nt)*dt

    Response = np.zeros((nt, Lx, Ly))
    Ns = np.zeros(nt) # count images per time step

    for tS in visual_stim['time_start']:
        new_time = times-tS
        cond = (new_time>(tstart-dt)) & (new_time<(tstop+dt))
        for ts, file in zip(new_time[cond], FILES[cond]):
            i0 = np.argmin((t-ts)**2)
            img = np.load(os.path.join(datafolder, 'ImagingCamera-imgs', file))
            # Response[i0,:,:] += gaussian_filter(img, smoothing)
            Response[i0,:,:] += img
            Ns[i0] +=1
    for i in range(nt):
        Response[i,:,:] /= Ns[i0]

    return t, Response


datafolder = os.path.expanduser(\
            '~/DATA/2025_12_18/17-22-17')
t, Ipsi = compute_resp(datafolder)
datafolder = os.path.expanduser(\
            '~/DATA/2025_12_18/18-16-29')
t, Contra = compute_resp(datafolder)



# %%

def compute_resp_map(Resp,
                     response_window = [1,2]):

    F0 = Resp[t<0,:,:].mean(axis=0)
    #
    cond = (t>response_window[0]) & (t<response_window[1])
    return (Resp[cond, :,:].mean(axis=0)-F0)/F0


ipsi_map = compute_resp_map(Ipsi)
contra_map = compute_resp_map(Contra)

bounds = [0, 0.1]

fig, AX = pt.figure(axes=(2,1), ax_scale=(1,1), wspace=0.3, right=5)

for ax, Map in zip(AX, [ipsi_map, contra_map]):
    im = ax.imshow(Map, cmap=pt.plt.cm.binary,
                    vmin=bounds[0], vmax=bounds[1])
    ax.axis('off')

fig.colorbar(im, ax=AX[1],
               shrink=0.9, aspect=10,
                label='mean $\Delta$F/F in [+1,+2]s ')

# %%
fig, ax = pt.figure()

W = 500
for Resp, map in zip([Ipsi, Contra], [ipsi_map, contra_map]):
    # i0, i1 = np.unravel_index(np.argmax(map), np.array(map).shape)
    # i0, i1 = 1000,1000
    # print(i0, i1)
    # ax.plot(t[1:], Resp[1:,i0-W:i0+W,i1-W:i1+W].mean(axis=(1,2)))

    ax.plot(t[1:50], Resp[1:50,:,:].mean(axis=(1,2)))
# %%
def plot_power_map(ax, fig, Map,
                   bounds=[0.,0.05]):

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

threshOD = 0.35

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


