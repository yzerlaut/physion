import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from scipy.signal import convolve, windows
from scipy.interpolate import interp1d

##############################################################################
# -------------------------------------------------------------------------- #
#          DEFAULT OPTIONS FOR FLUORESCENCE (dFoF) PROCESSING                # 

ROI_TO_NEUROPIL_INCLUSION_FACTOR = 1.0 # ratio to discard ROIs with weak fluo compared to neuropil
ROI_TO_NEUROPIL_INCLUSION_FACTOR_METRIC = 'mean' # either 'mean' or 'std'
METHOD = 'percentile' # either 'minimum', 'percentile', 'sliding_minimum', 'sliding_percentile', 'hamming', 'sliding_minmax'
T_SLIDING = 300. # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile' | 'hamming' | 'sliding_minmax')
PERCENTILE = 10. # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile' | 'hamming')
NEUROPIL_CORRECTION_FACTOR = 0.8 # fraction of neuropil substracted to fluorescence

# -------------------------------------------------------------------------- #
##############################################################################

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
        Flow[:,sbsmplIndices] = gaussian_filter1d(Flow[:,sbsmplIndices], 
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
        Flow = gaussian_filter1d(array, pre_smoothing)
    else:
        Flow = array

    Flow = minimum_filter1d(Flow, Window, mode='reflect')

    if with_smoothing:
        Flow = gaussian_filter1d(Flow, Window, axis=1)

    return Flow

def compute_sliding_minmax(array, Window,
                           pre_smoothing=0): 
    if pre_smoothing>0:
        Flow = gaussian_filter1d(array, pre_smoothing)
    else:
        Flow = array

    Flow = minimum_filter1d(Flow, Window, mode='reflect')
    Flow = maximum_filter1d(Flow, Window, mode='reflect')

    return Flow

def compute_hamming(array, Window, percentile) :
    Flow = []
    hamming_window = windows.hamming(Window)

    for i in range(len(array)):
        F_smooth = convolve(array[i], hamming_window, mode='same') / sum(hamming_window)
        roi_percentile = np.percentile(F_smooth, percentile)
        F_below_percentile = np.extract(F_smooth <= roi_percentile, F_smooth)
        f0 = np.mean(F_below_percentile)
        f0 = [f0]*len(array[i])
        Flow.append(f0)
    
    return np.array(Flow)

def compute_F0(F, dt,
               method=METHOD,
               percentile=PERCENTILE,
               sliding_window=T_SLIDING):
    """
    baseline F0 of the fluorescence F, shape (nROIs, nTimes)

    dt: time interval between the samples of F (s), converts the
            sliding window (s) into a number of samples
    """

    if method=='minimum':
        return compute_minimum(F)

    elif method=='percentile':
        return compute_percentile(F, percentile=percentile)

    elif method=='sliding_minimum':
        return compute_sliding_minimum(F,
                                       int(sliding_window/dt),
                                       with_smoothing=True)

    elif method=='sliding_percentile':
        return compute_sliding_percentile(F, percentile,
                                          int(sliding_window/dt),
                                          with_smoothing=True)

    elif method=='hamming':
        return compute_hamming(F, int(sliding_window/dt), percentile)
    
    elif method=='sliding_minmax':
        return compute_sliding_minmax(F, int(sliding_window/dt), 60)
    
    else:
        print('\n --- method not recognized --- \n ')


def compute_neuropil_factor(F, Fneu,
                            neuropil_percentile_range=(10, 90),
                            fluo_percentile_for_regression=10,
                            nBins=16):
    """
    neuropil contamination factor of each ROI: slope of the lower envelope of
        the fluorescence F as a function of the neuropil Fneu

        - the neuropil is binned by percentiles: nBins bins between the
            percentiles "neuropil_percentile_range" of each ROI's neuropil
        - in each bin: the "fluo_percentile_for_regression" percentile of F,
            paired with the neuropil value at the upper edge of the bin
        - linear regression (least squares) of these points -> slope

    F, Fneu: arrays of shape (nROIs, nTimes)
    returns the slopes, shape (nROIs,)
    """
    F, Fneu = np.asarray(F, dtype=float), np.asarray(Fneu, dtype=float)
    nTimes = F.shape[1]

    # samples of each ROI sorted by neuropil value: the percentile bins of
    #   the neuropil are then the same slices of time samples for all ROIs
    order = np.argsort(Fneu, axis=1)
    F_sorted = np.take_along_axis(F, order, axis=1)

    # bin edges, as positions in the sorted samples (np.percentile convention)
    q = np.linspace(*neuropil_percentile_range, nBins+1)
    edges = np.ceil(q/100.*(nTimes-1)).astype(int)

    # lower envelope of F in each neuropil bin
    y = np.stack([np.percentile(F_sorted[:, i0:i1],
                                fluo_percentile_for_regression, axis=1)
                  for i0, i1 in zip(edges[:-1], edges[1:])], axis=1)
    # neuropil value at the upper edge of each bin
    x = np.percentile(Fneu, q[1:], axis=1).T

    # least-squares slope of y vs x, for all ROIs
    dx = x-x.mean(axis=1, keepdims=True)
    dy = y-y.mean(axis=1, keepdims=True)
    return np.sum(dx*dy, axis=1)/np.sum(dx**2, axis=1)
