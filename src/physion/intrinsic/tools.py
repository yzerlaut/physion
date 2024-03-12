import os, sys, pathlib, pynwb, itertools, skimage
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colorbar, colors
from skimage import measure
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from PIL import Image

from physion.utils import plot_tools as pt

# from datavyz import graph_env
ge_screen = None

default_segmentation_params={'phaseMapFilterSigma': 2.,
                             'signMapFilterSigma': 9.,
                             'signMapThr': 0.35,
                             'eccMapFilterSigma': 10.,
                             'splitLocalMinCutStep': 5.,
                             'mergeOverlapThr': 0.1,
                             'closeIter': 3,
                             'openIter': 3,
                             'dilationIter': 15,
                             'borderWidth': 1,
                             'smallPatchThr': 100,
                             'visualSpacePixelSize': 0.5,
                             'visualSpaceCloseIter': 15,
                             'splitOverlapThr': 1.1}

def load_maps(datafolder, Nsubsampling=4):

    if os.path.isfile(os.path.join(datafolder, 'metadata.npy')):
        print('\n  loading previously calculated maps --> can be overwritten un the UI ! \n ')
        metadata= np.load(os.path.join(datafolder, 'metadata.npy'),
                       allow_pickle=True).item()
        if 'Nsubsampling' in metadata:
            Nsubsampling = metadata['Nsubsampling']
    else:
        metadata = None
    
    if os.path.isfile(os.path.join(datafolder, 'raw-maps.npy')):
        print('\n  loading previously calculated maps --> can be overwritten un the UI ! \n ')
        maps = np.load(os.path.join(datafolder, 'raw-maps.npy'),
                       allow_pickle=True).item()
    else:
        maps = {}

    if os.path.isfile(os.path.join(datafolder, 'vasculature-%s.tif' %metadata['subject'])):
        maps['vasculature'] = np.array(Image.open(os.path.join(datafolder,\
                'vasculature-%s.tif' %metadata['subject'])))
        maps['vasculature'] = maps['vasculature'][::Nsubsampling,::Nsubsampling]
    elif os.path.isfile(os.path.join(datafolder, 'vasculature-%s.npy' %metadata['subject'])):
        maps['vasculature'] = np.load(os.path.join(datafolder,\
                'vasculature-%s.npy' %metadata['subject']))
        maps['vasculature'] = maps['vasculature'][::Nsubsampling,::Nsubsampling]
    elif os.path.isfile(os.path.join(datafolder, 'vasculature.npy')):
        maps['vasculature'] = np.load(os.path.join(datafolder, 'vasculature.npy'))

    if os.path.isfile(os.path.join(datafolder, 'fluorescence-%s.tif' %metadata['subject'])):
        maps['fluorescence'] = np.array(Image.open(os.path.join(datafolder,\
                'fluorescence-%s.tif' %metadata['subject'])))
        maps['fluorescence'] = maps['fluorescence'][::Nsubsampling,::Nsubsampling]
    elif os.path.isfile(os.path.join(datafolder, 'fluorescence-%s.npy' %metadata['subject'])):
        maps['fluorescence'] = np.load(os.path.join(datafolder,\
                'fluorescence-%s.npy' %metadata['subject']))
        maps['fluorescence'] = maps['fluorescence'][::Nsubsampling,::Nsubsampling]
    elif os.path.isfile(os.path.join(datafolder, 'fluorescence.npy')):
        maps['fluorescence'] = np.load(os.path.join(datafolder, 'fluorescence.npy'))


    return maps


def resample_data(array, old_time, time):
    new_array = 0*time
    for i1, i2 in zip(range(len(time)-1), range(1, len(time))):
        cond=(old_time>time[i1]) & (old_time<=time[i2])
        if len(cond)>1:
            new_array[i1] = np.mean(array[cond])
        elif len(cond)==1:
            new_array[i1] = array[cond][0]
    return new_array


def resample_img(img, Nsubsampling):
    if Nsubsampling>1:
        if len(img.shape)==3:
            # means movie !
            return measure.block_reduce(img, block_size=(1,
                                                         Nsubsampling,
                                                         Nsubsampling), func=np.mean)

        else:
            return measure.block_reduce(img, block_size=(Nsubsampling,
                                                     Nsubsampling), func=np.mean)
    else:
        return img


def load_single_datafile(datafile):
    """
    the image data need interpolation to get regularly spaced data for FFT
    """
    io = pynwb.NWBHDF5IO(datafile, 'r')
    nwbfile = io.read()
    t, x = nwbfile.acquisition['image_timeseries'].timestamps[:].astype(np.float64),\
        nwbfile.acquisition['image_timeseries'].data[:,:,:].astype(np.uint16)
    interp_func = interp1d(t, x, axis=0, kind='nearest', fill_value='extrapolate')
    real_t = nwbfile.acquisition['angle_timeseries'].timestamps[:]
    io.close()
    return real_t, interp_func(real_t)
    # return t, nwbfile.acquisition['image_timeseries'].data[:,:,:]

def load_raw_data(datafolder, protocol,
                  run_id='sum'):

    params = np.load(os.path.join(datafolder, 'metadata.npy'),
                     allow_pickle=True).item()

    if run_id=='sum':
        Data, n = None, 0
        for i in range(1, 15): # no more than 15 repeats...(but some can be removed, hence the "for" loop)
            if os.path.isfile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i))):
                t, data  = load_single_datafile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i)))
                if Data is None:
                    Data = data
                    n = 1
                else:
                    Data += data
                    n+=1
        if n>0:
            return params, (t, Data/n)
        else:
            return params, (None, None)

    elif os.path.isfile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id))):
        return params, load_single_datafile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id)))
    else:
        print('"%s" file not found' % os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id)))


def preprocess_data(data, Facq,
                    temporal_smoothing=0,
                    spatial_smoothing=0,
                    high_pass_filtering=0):

    pData = resample_img(data, spatial_smoothing) # pre-processed data

    if high_pass_filtering>0:
        pData = butter_highpass_filter(pData-pData.mean(axis=0), high_pass_filtering, Facq, axis=0)
    if temporal_smoothing>0:
        pData = gaussian_filter1d(pData, Facq*temporal_smoothing, axis=0)

    return pData

def perform_fft_analysis(data, nrepeat,
                         phase_range='-pi:pi'):
    """
    Fourier transform
        we center the phase around pi/2
    """
    spectrum = np.fft.fft(-data, axis=0)

    # relative power w.r.t. luminance
    rel_power = np.abs(spectrum)[nrepeat, :, :]/data.shape[0]/data.mean(axis=0)

    if phase_range=='-pi:pi':
        phase = np.angle(spectrum)[nrepeat, :, :]
    elif phase_range=='0:2*pi':
        phase = (np.angle(spectrum)[nrepeat, :, :])%(2.*np.pi)

    return rel_power, phase


def compute_phase_power_maps(datafolder, direction,
                             maps={},
                             p=None, t=None, data=None,
                             run_id='sum',
                             phase_range='-pi:pi'):

    # load raw data
    if (p is None) or (t is None) or (data is None):
        p, (t, data) = load_raw_data(datafolder, direction, run_id=run_id)

    if 'vasculature' not in maps:
        if os.path.isfile(os.path.join(datafolder, 'vasculature.npy')):
            maps['vasculature'] = np.load(os.path.join(datafolder, 'vasculature.npy'))
        else:
            maps['vasculature'] = np.zeros((data.shape[1], data.shape[2]))


    # FFT and write maps
    maps['%s-power' % direction],\
           maps['%s-phase' % direction] = perform_fft_analysis(data, p['Nrepeat'],
                                                    phase_range=phase_range)

    return maps

def get_phase_to_angle_func(datafolder, direction):
    """
    converti stimulus phase to visual angle
    """

    p= np.load(os.path.join(datafolder, 'metadata.npy'),
                     allow_pickle=True).item()

    # phase to angle conversion
    if direction=='up':
        bounds = [p['STIM']['zmin'], p['STIM']['zmax']]
    elif direction=='right':
        bounds = [p['STIM']['xmin'], p['STIM']['xmax']]
    elif direction=='down':
        bounds = [p['STIM']['zmax'], p['STIM']['zmin']]
    else:
        bounds = [p['STIM']['xmax'], p['STIM']['xmin']]

    # keep phase to angle relathionship    /!\ [-PI/2, 3*PI/2] interval /!\
    phase_to_angle_func = lambda x: bounds[0]+\
                    (x+np.pi/2)/(2*np.pi)*(bounds[1]-bounds[0])

    return phase_to_angle_func


def compute_retinotopic_maps(datafolder, map_type,
                             maps={}, # we fill the dictionary passed as argument
                             altitude_Zero_shift=10,
                             azimuth_Zero_shift=60,
                             run_id='sum',
                             keep_maps=False,
                             verbose=True,
                             phase_shift=0):
    """
    map type is either "altitude" or "azimuth"
    """

    if verbose:
        print('- computing "%s" retinotopic maps [...] ' % map_type)

    if map_type=='altitude':
        directions = ['up', 'down']
        phase_to_angle_func = get_phase_to_angle_func(datafolder, 'up')
    else:
        directions = ['right', 'left']
        phase_to_angle_func = get_phase_to_angle_func(datafolder, 'right')

    for direction in directions:
        if (('%s-power'%direction) not in maps) and not keep_maps:
            compute_phase_power_maps(datafolder, direction,
                                     maps=maps,
                                     phase_shift=phase_shift)

    if verbose:
        print('-> retinotopic map calculation over ! ')

    # build maps
    maps['%s-power' % map_type] = .5*(maps['%s-power' % directions[0]]+\
                                      maps['%s-power' % directions[1]])

    maps['%s-delay' % map_type] = 0.5*(maps['%s-phase' % directions[0]]+\
                                       maps['%s-phase' % directions[1]])

    maps['%s-phase-diff' % map_type] = (maps['%s-phase' % directions[0]]-
                                        maps['%s-phase' % directions[1]])

    maps['%s-retinotopy' % map_type] = phase_to_angle_func(\
                        maps['%s-phase-diff' % map_type])

    return maps


def build_trial_data(maps,
                     subject='',
                     comments='',
                     dateRecorded='2022-01-01',
                     with_params=False):
    """
    prepare the data to be saved
    """

    output = {'mouseID':subject,
              'comments':comments,
              'dateRecorded':dateRecorded}

    for key1, key2 in zip(\
            ['vasculature', 'altitude-retinotopy', 'azimuth-retinotopy',\
                            'altitude-power', 'azimuth-power'],
            ['vasculature', 'altPos', 'aziPos', 'altPower', 'aziPower']):
        if key1 in maps:
            output[key2+'Map'] = maps[key1]
        else:
            output[key2+'Map'] = 0.*maps['vasculature']

    if with_params:
        if 'params' in maps:
            output['params']=maps['params']
        else:
            output['params']=default_segmentation_params

    return output

# -------------------------------------------------------------- #
# ----------- PLOT FUNCTIONS ----------------------------------- #
# -------------------------------------------------------------- #

def add_scale_bar(ax, height=2.7, color='r'):

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    dx, dy = xlim[1]-xlim[0], ylim[1]-ylim[0]
    x0 = xlim[0]+0.02*dx
    y1 = ylim[1]-0.02*dy

    ax.plot([x0,x0], [y1, y1-dy/height], color=color, lw=1)
    ax.annotate('1mm', (x0+0.01*dx, y1),
                ha='left', va='top', rotation=90, color=color, fontsize=6)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

def add_arrow(ax, angle,
              lw=0.3,
              fontsize=6):

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    dx, dy = np.abs(xlim[1]-xlim[0]), np.abs(ylim[1]-ylim[0])

    start = (xlim[0]+dx/2, ylim[1])
    delta = (np.sin(angle/180.*np.pi)*dy, dy)

    ax.annotate('Anterior ', start,
                ha='right', va='top', color='r', fontsize=6)
    ax.annotate('Posterior  ', (start[0]+0.99*delta[0], start[1]+0.99*delta[1]),
                ha='right', va='bottom', color='r', fontsize=fontsize)
    ax.arrow(*start, *delta, color='r', lw=lw)

    start = (xlim[1], ylim[1]+dy/2)
    delta = (-dx, np.sin(angle/180.*np.pi)*dx)
    
    ax.annotate('Lateral ', start,
                ha='right', color='r', fontsize=fontsize)
    ax.arrow(*start, *delta, color='r', lw=lw)
    ax.annotate(' Medial', (start[0]+0.99*delta[0], start[1]+0.99*delta[1]),
                ha='left', va='top', color='r', fontsize=fontsize)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_phase_map(ax, fig, Map,
                   phase_range='-pi:pi'):
    if phase_range=='-pi:pi':
        im = ax.imshow(Map,
                       cmap=plt.cm.twilight, vmin=-np.pi, vmax=np.pi)
        cbar = fig.colorbar(im, ax=ax,
                            ticks=[-np.pi, 0, np.pi], 
                            shrink=0.4,
                            aspect=10,
                            label='phase (Rd)')
        cbar.ax.set_yticklabels(['-$\pi$', '0', '$\pi$'])
    else:
        im = ax.imshow(Map,
                       cmap=plt.cm.twilight, vmin=0, vmax=2*np.pi)
        cbar = fig.colorbar(im, ax=ax,
                            ticks=[0, np.pi, 2*np.pi], 
                            shrink=0.4,
                            aspect=10,
                            label='phase (Rd)')
        cbar.ax.set_yticklabels(['0', '$\pi$', '2$\pi$'])

def plot_power_map(ax, fig, Map,
                   bounds=None):

    if bounds is None:
        bounds = [np.min(1e4*Map), np.max(1e4*Map)]

    im = ax.imshow(1e4*Map, cmap=plt.cm.binary,
                   vmin=bounds[0], vmax=bounds[1])
    ax.set_title('power map')
    fig.colorbar(im, ax=ax,
                 shrink=0.4,
                 aspect=10,
                 label='relative power \n ($10^{-4}$ a.u.)')


def plot_phase_power_maps(maps, direction,
                          phase_range='-pi:pi'):

    fig, AX = plt.subplots(1, 2, figsize=(7,2.3))
    plt.subplots_adjust(bottom=0, top=1, wspace=1, right=0.8)

    plt.annotate('"%s" protocol' % direction, (0.5,.99), ha='center', va='top',
                 xycoords='figure fraction')

    # # power first
    plot_power_map(AX[0], fig, maps['%s-power' % direction])
    
    # # then phase of the stimulus
    plot_phase_map(AX[1], fig, maps['%s-phase' % direction],
                   phase_range=phase_range)

    for ax in AX:
        ax.axis('off')

    return fig

def plot_retinotopic_maps(maps, map_type='altitude',
                          max_retinotopic_angle=80,
                          ge=ge_screen):
    
    if map_type=='altitude':
        plus, minus = 'up', 'down'
    else:
        plus, minus = 'left', 'right'
        
    fig, AX = plt.subplots(3, 2, figsize=(3.9,3.9))
    plt.subplots_adjust(bottom=0.05, left=0.05, hspace=.5, wspace=0.5, right=0.8)

    plt.annotate('"%s" maps' % map_type, (0.5,.99), ha='center', va='top', 
                 xycoords='figure fraction', size='small')
    
    plot_phase_map(AX[0][0], fig, maps['%s-phase' % plus])
    plot_phase_map(AX[0][1], fig, maps['%s-phase' % minus])

    AX[0][0].annotate('$\phi$+', (1,1), ha='right', va='top', color='w', xycoords='axes fraction')
    AX[0][1].annotate('$\phi$-', (1,1), ha='right', va='top', color='w', xycoords='axes fraction')
    AX[0][0].set_title('phase map: "%s"' % plus)
    AX[0][1].set_title('phase map: "%s"' % minus)

    bounds = [1e4*np.min([maps['%s-power' % x].min() for x in [plus, minus]]),
              1e4*np.max([maps['%s-power' % x].max() for x in [plus, minus]])]

    plot_power_map(AX[1][0], fig, maps['%s-power' % plus], bounds=bounds)
    AX[1][0].set_title('power map: "%s"' % plus)

    plot_power_map(AX[1][1], fig, maps['%s-power' % minus], bounds=bounds)
    AX[1][1].set_title('power map: "%s"' % minus)
    
    # bounds = [np.min(maps['%s-retinotopy' % map_type]),
              # np.max(maps['%s-retinotopy' % map_type])]
    bounds = [-max_retinotopic_angle, max_retinotopic_angle]
    
    im = AX[2][0].imshow(maps['%s-delay' % map_type], cmap=plt.cm.twilight,\
                    vmin=-np.pi/2, vmax=3*np.pi/2)
    fig.colorbar(im, ax=AX[2][0])
    AX[2][0].annotate('$\phi^{+}$+$\phi^{-}$', (0,1),
            ha='right', va='top', rotation=90, xycoords='axes fraction')
    AX[2][0].set_title('(hemodynamic)\ndelay map')

    im = AX[2][1].imshow(maps['%s-retinotopy' % map_type], cmap=plt.cm.PRGn,\
                    vmin=bounds[0], vmax=bounds[1])
    fig.colorbar(im, ax=AX[2][1],
                 label='angle (deg.)\n visual field')
    AX[2][1].annotate('F[$\phi^{+}$-$\phi^{-}$]', (0,1),
            ha='right', va='top', rotation=90, xycoords='axes fraction')
    AX[2][1].set_title('retinotopy map')

    for Ax in AX:
        for ax in Ax:
            ax.axis('off')
        
    return fig


def add_patches(trial, ax):

    signMapf = trial.signMapf
    rawPatchMap = trial.rawPatchMap
    
    patchMapDilated = RetinotopicMapping.dilationPatches2(rawPatchMap,\
            dilationIter=trial.params['dilationIter'],
            borderWidth=trial.params['borderWidth'])

    rawPatches = RetinotopicMapping.labelPatches(patchMapDilated, signMapf)

    rawPatches = RetinotopicMapping.sortPatches(rawPatches)

    for key, currPatch in rawPatches.items():

        ax.imshow(currPatch.getSignedMask(),\
                  vmax=1, vmin=-1, interpolation='nearest', alpha=0.5, cmap='jet')


def save_maps(maps, filename):
    """ removes the functions from the maps to be able to save """
    Maps = {}
    for m in maps:
        if 'func' not in m:
            Maps[m] = maps[m]

    np.save(filename, Maps)
