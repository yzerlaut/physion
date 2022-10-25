import numpy as np
from scipy.ndimage import filters
import time

####################################
# ---------------------------------
# DEFAULT_CA_IMAGING_OPTIONS

METHOD = 'maximin' # either 'maximin' or 'sliding_percentile'
T_SLIDING_MIN = 60. # seconds
PERCENTILE_SLIDING_MIN = 5. # percent
NEUROPIL_CORRECTION_FACTOR = 0.7

# ---------------------------------
####################################

def fill_center_and_edges(N, Window, smv):
    sliding_min = np.zeros(N)
    iw = int(Window/2)+1
    if len(smv.shape)==1:
        sliding_min[:iw] = smv[0]
        sliding_min[-iw:] = smv[-1]
        sliding_min[iw:iw+smv.shape[-1]] = smv
    elif len(smv.shape)==2:
        sliding_min[:,:iw] = np.broadcast_to(smv[:,0], (iw, array.shape[0])).T
        sliding_min[:,-iw:] = np.broadcast_to(smv[:,-1], (iw, array.shape[0])).T
        sliding_min[:,iw:iw+smv.shape[-1]] = smv
    return sliding_min

# numpy code for ~efficiently evaluating the distrib percentile over a sliding window
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


def sliding_percentile(array, percentile, Window,
                       with_smoothing=False):

    x = np.zeros(len(array))
    y0 = strided_app(array, Window, 1)
    
    y = np.percentile(y0, percentile, axis=-1)
    
    x[:int(Window/2)] = y[0]
    x[int(Window/2):int(Window/2)+len(y)] = y
    x[-int(Window/2):] = y[-1]
    if with_smoothing:
        return gaussian_filter1d(x, Window)
    else:
        return x
    
def compute_sliding_percentile(array, percentile, Window,
                               with_smoothing=False):
    """
    trying numpy code to evaluate efficiently the distrib percentile over a sliding window
    making use of "stride tricks" for fast looping over the sliding window
    
        not really efficient so far... :(
    
    see: 
    https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    """
    
    # using a sliding "view" of the array
    sliding_percentile = np.zeros(array.shape)
    for roi in range(array.shape[0]):
        view = np.lib.stride_tricks.sliding_window_view(array[roi,:], Window, axis=-1)
        smv = np.percentile(view, percentile, axis=-1)
        # replacing values, N.B. need to deal with edges
        sliding_percentile[roi,:] = fill_center_and_edges(len(sliding_percentile[roi,:]), Window, smv)
        if with_smoothing:
            sliding_percentile[roi,:] = filters.gaussian_filter1d(sliding_percentile[roi,:], Window, axis=-1)

    return sliding_percentile


def compute_sliding_minmax(array, Window, sig=2):
    if sig>0:
        Flow = filters.gaussian_filter1d(array, sig)
    else:
        Flow = array
    Flow = filters.minimum_filter1d(Flow, Window, mode='wrap')
    Flow = filters.maximum_filter1d(Flow, Window, mode='wrap')
    return Flow


def compute_sliding_F0(data, F,
                       method=METHOD,
                       percentile=PERCENTILE_SLIDING_MIN,
                       sliding_window=T_SLIDING_MIN):
    if method in ['maximin', 'minmax']:
        return compute_sliding_minmax(F,
                                      int(sliding_window/data.CaImaging_dt))
    elif 'percentile' in method:
        # return compute_sliding_percentile(F, sliding_percentile,
        #                                   int(sliding_window/data.CaImaging_dt)_
        # GOING BACK TO OLD SIMPLE CODE (for loop over rois):
        F0 = np.zeros(F.shape)
        for i in range(F0.shape[0]):
            F0[i,:] = sliding_percentile(F[i,:], percentile,
                                         int(sliding_window/data.CaImaging_dt))
        return F0
    else:
        print('\n --- method not recognized --- \n ')
        


def compute_dFoF(data,  
                 neuropil_correction_factor=NEUROPIL_CORRECTION_FACTOR,
                 method_for_F0=METHOD,
                 percentile=PERCENTILE_SLIDING_MIN,
                 sliding_window=T_SLIDING_MIN,
                 return_corrected_F_and_F0=False,
                 verbose=True):
    """
    compute fluorescence variation with a neuropil correction set by the 
    factor neuropil_correction_factor
    """

    
    if verbose:
        tick = time.time()
        print('\ncalculating dF/F with method "%s" [...]' % method_for_F0)
        
    if (neuropil_correction_factor>1) or (neuropil_correction_factor<0):
        print('/!\ neuropil_correction_factor has to be in the interval [0.,1]')
        print('neuropil_correction_factor set to 0 !')
        neuropil_correction_factor=0.

    # ## ------------------------------------ ##
    # ############# classic way ################
    # ## ------------------------------------ ##
    # # performing correction 
    # F = data.Fluorescence.data[:,:]-neuropil_correction_factor*data.Neuropil.data[:,:]
        
    # F0 = compute_sliding_F0(data, F,
    #                         method=method_for_F0,
    #                         percentile=percentile,
    #                         sliding_window=sliding_window)

    # # exclude cells with negative F0
    # valid_roiIndices = (np.min(F0, axis=1)>0) & (F0.mean(axis=1)>F.std(axis=1))

    # if verbose:
    #     if np.sum(~valid_roiIndices)>0:
    #         print('\n  ** %i ROIs were discarded with the positive F0 criterion (%.1f%%) ** \n'\
    #               % (np.sum(~valid_roiIndices), 100*np.sum(~valid_roiIndices)/F0.shape[0]))
    #     else:
    #         print('\n  ** all ROIs passed the positive F0 criterion ** \n')
            
    # data.nROIs = np.sum(valid_roiIndices)
    # data.dFoF = (F[valid_roiIndices,:]-F0[valid_roiIndices,:])/F0[valid_roiIndices,:]
    # data.valid_roiIndices = np.arange(data.iscell.sum())[valid_roiIndices]

    ## ----------------------------------------------------------- ##
    ############# simple way to insure F0 far from 0 ################
    ## ----------------------------------------------------------- ##

    # F0 on raw fluorescence (uncorrected)
    F0 = compute_sliding_F0(data, data.rawFluo,
                            method=method_for_F0,
                            percentile=percentile,
                            sliding_window=sliding_window)

    dF = data.rawFluo-neuropil_correction_factor*data.neuropil

    # exclude cells with negative dF
    valid_roiIndices = (np.mean(dF, axis=1)>0)

    if verbose:
        if np.sum(~valid_roiIndices)>0:
            print('\n  ** %i ROIs were discarded with the positive F0 criterion (%.1f%%) ** \n'\
                  % (np.sum(~valid_roiIndices), 100*np.sum(~valid_roiIndices)/F0.shape[0]))
        else:
            print('\n  ** all ROIs passed the positive F0 criterion ** \n')
            
    data.dFoF = dF[valid_roiIndices,:]/F0[valid_roiIndices,:]
    data.valid_roiIndices = np.arange(data.iscell.sum())[valid_roiIndices]
    
    if verbose:
        print('-> dFoF calculus done !  (calculation took %.1fs)' % (time.time()-tick))
    
    if return_corrected_F_and_F0:
        return dF[valid_roiIndices,:], F0[valid_roiIndices,:]
    else:
        return None

    
########################################################################################    
####################### old code, should be deprecated #################################
########################################################################################    




def compute_CaImaging_raster(data, CaImaging_key,
                             roiIndices='all',
                             normalization='None',
                             compute_CaImaging_options=dict(T_sliding_min=T_SLIDING_MIN,
                                                            percentile_sliding_min=PERCENTILE_SLIDING_MIN),
                             verbose=False):
    """
    normalization can be: 'None', 'per line'

    """

    if (not type(roiIndices) in [list, np.array]) and (roiIndices=='all'):
        roiIndices = np.arange(data.iscell.sum())

    if verbose:
        print('computing raster [...]')
    raster = compute_CaImaging_trace(data, CaImaging_key, roiIndices, **compute_CaImaging_options)

    if verbose:
        print('normalizing raster [...]')
    if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
        for n in range(raster.shape[0]):
            Fmax, Fmin = raster[n,:].max(), raster[n,:].min()
            if Fmax>Fmin:
                raster[n,:] = (raster[n,:]-Fmin)/(Fmax-Fmin)

    return raster
              







        
