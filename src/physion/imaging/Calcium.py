import numpy as np
from scipy.ndimage import filters
from scipy.interpolate import interp1d
import time

####################################
# ---------------------------------
# DEFAULT_CA_IMAGING_OPTIONS

ROI_TO_NEUROPIL_INCLUSION_FACTOR = 1.1 # ratio to discard ROIs with weak fluo compared to neuropil
METHOD = 'percentile' # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
T_SLIDING = 300. # seconds
PERCENTILE = 10. # percent
NEUROPIL_CORRECTION_FACTOR = 0.8

# ---------------------------------
####################################

def compute_minimum(array):
    return np.repeat(np.min(array, axis=1)[:,np.newaxis],
                     array.shape[1],
                     axis=1)

def compute_percentile(array, percentile):
    return np.repeat(np.percentile(array, percentile, axis=1)[:,np.newaxis],
                     array.shape[1],
                     axis=1)

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


def compute_sliding_minimum(array, Window,
                            pre_smoothing=0,
                            with_smoothing=False):
    if pre_smoothing>0:
        Flow = filters.gaussian_filter1d(array, pre_smoothing)
    else:
        Flow = array

    Flow = filters.minimum_filter1d(Flow, Window, mode='wrap')

    if with_smoothing:
        Flow = filters.gaussian_filter1d(Flow, Window, axis=1)

    return Flow

def compute_F0(data, F,
               method=METHOD,
               percentile=PERCENTILE,
               sliding_window=T_SLIDING):

    if method=='minimum':
        return compute_minimum(F)

    elif method=='percentile':
        return compute_percentile(F, percentile=percentile)

    elif method=='sliding_minimum':
        return compute_sliding_minimum(F,
                                       int(sliding_window/data.CaImaging_dt),
                                       with_smoothing=True)

    elif method=='sliding_percentile':
        return compute_sliding_percentile(F, percentile,
                                          int(sliding_window/data.CaImaging_dt),
                                          with_smoothing=True)

    else:
        print('\n --- method not recognized --- \n ')
        


def compute_dFoF(data,  
                 roi_to_neuropil_fluo_inclusion_factor=ROI_TO_NEUROPIL_INCLUSION_FACTOR,
                 neuropil_correction_factor=NEUROPIL_CORRECTION_FACTOR,
                 method_for_F0=METHOD,
                 percentile=PERCENTILE,
                 sliding_window=T_SLIDING,
                 with_correctedFluo_and_F0=False,
                 smoothing=None,
                 verbose=True):
    """
    -----------------
    Compute the *Delta F over F* quantity
    -----------------
    1) substract a fraction of the neuropil component to get the corrected fluo: cF
        - with the "neuropil_correction_factor" parameter
    2) restrict to ROIs that have a mean fluorescence larger that the mean neuropil
        - with the "roi_to_neuropil_fluo_inclusion_factor" parameter
            the link with the original ROIs are through: data.valid_ROIs & data.unvalid_ROIs
    3)  determine the sliding baseline component: cF0
        - with the "method" parameter, method can be either: maximin / sliding_percentile
        - with the "percentile" parameter (in percent)
        - with the "sliding_windows" parameter (in s)
    4) copmutes the ratio between (cF-cF0)/cF0
    5) [optional] adds a Gaussian smoothing (smoothing in frame units)
    """

    if verbose:
        tick = time.time()
        print('\ncalculating dF/F with method "%s" [...]' % method_for_F0)
        
    if (neuropil_correction_factor>1) or (neuropil_correction_factor<0):
        print('/!\ neuropil_correction_factor has to be in the interval [0.,1]')
        print('neuropil_correction_factor set to 0 !')
        neuropil_correction_factor=0.

    #######################################################################

    # Step 1) ->  performing neuropil correction 
    correctedFluo = data.rawFluo-\
            neuropil_correction_factor*data.neuropil
    # correctedFluo = data.rawFluo[valid_roiIndices, :]-\
            # neuropil_correction_factor*data.neuropil[valid_roiIndices, :]
    
    # Step 2) -> compute the F0 term (~ sliding minimum/percentile)
    correctedFluo0 = compute_F0(data, correctedFluo,
                                method=method_for_F0,
                                percentile=percentile,
                                sliding_window=sliding_window)

    # Step 3) -> determine the valid ROIs
    # ROIs with strictly positive baseline
    valid_roiIndices = np.min(correctedFluo0, axis=1)>1
    # valid_roiIndices = (\
            # (np.mean(data.rawFluo, axis=1)>\
            # roi_to_neuropil_fluo_inclusion_factor*np.mean(data.neuropil, axis=1)))

    # Step 4) -> compute the delta F over F quantity: dFoF = (F-F0)/F0
    data.dFoF = (correctedFluo[valid_roiIndices, :]-\
      correctedFluo0[valid_roiIndices, :])/correctedFluo0[valid_roiIndices, :]

    # Step 5) -> Gaussian smoothing if required
    if smoothing is not None:
        data.dFoF = filters.gaussian_filter1d(data.dFoF, smoothing, axis=1)

    #######################################################################
    if verbose:
        if np.sum(~valid_roiIndices)>0:
            print('\n  ** %i ROIs were discarded with the positive F0 criterion (%.1f%%) ** \n'\
                  % (np.sum(~valid_roiIndices),
                      100*np.sum(~valid_roiIndices)/correctedFluo.shape[0]))
        else:
            print('\n  ** all ROIs passed the positive F0 criterion ** \n')
            
    # we update the previous quantities
    data.initialize_ROIs(\
            valid_roiIndices= np.arange(data.iscell.sum())[valid_roiIndices])

    # we resrict the rawFluo and neuropil to valid ROIs
    data.rawFluo = data.rawFluo[valid_roiIndices,:]
    data.neuropil = data.neuropil[valid_roiIndices,:]

    if with_correctedFluo_and_F0:
        data.correctedFluo0 = correctedFluo0
        data.correctedFluo = correctedFluo
    
    if verbose:
        print('-> dFoF calculus done !  (calculation took %.1fs)' % (time.time()-tick))

    return None
