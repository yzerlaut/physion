import numpy as np
from scipy.ndimage import filters
import time

####################################
# ---------------------------------
# DEFAULT_CA_IMAGING_OPTIONS

ROI_TO_NEUROPIL_INCLUSION_FACTOR = 1.1 # ratio to discard ROIs with weak fluo compared to neuropil
METHOD = 'maximin' # either 'maximin' (for speed) or 'sliding_percentile'
T_SLIDING_MIN = 300. # seconds
PERCENTILE_SLIDING_MIN = 10. # percent
NEUROPIL_CORRECTION_FACTOR = 0.8

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
                 roi_to_neuropil_fluo_inclusion_factor=ROI_TO_NEUROPIL_INCLUSION_FACTOR,
                 neuropil_correction_factor=NEUROPIL_CORRECTION_FACTOR,
                 method_for_F0=METHOD,
                 percentile=PERCENTILE_SLIDING_MIN,
                 sliding_window=T_SLIDING_MIN,
                 with_correctedFluo_and_F0=False,
                 verbose=True):
    """
    -----------------
    Compute the *Delta F over F* quantity
    -----------------
    1) restrict to ROIs that have a mean fluorescence larger that the mean neuropil
        - with the "roi_to_neuropil_fluo_inclusion_factor" parameter
            the link with the original ROIs are through: data.valid_ROIs & data.unvalid_ROIs
    2) substract a fraction of the neuropil component to get the corrected fluo: cF
        - with the "neuropil_correction_factor" parameter
    3)  determine the sliding baseline component: cF0
        - with the "method" parameter, method can be either: maximin / sliding_percentile
        - with the "percentile" parameter (in percent)
        - with the "sliding_windows" parameter (in s)
    4) copmutes the ratio between (cF-cF0)/cF0
    """

    if verbose:
        tick = time.time()
        print('\ncalculating dF/F with method "%s" [...]' % method_for_F0)
        
    if (neuropil_correction_factor>1) or (neuropil_correction_factor<0):
        print('/!\ neuropil_correction_factor has to be in the interval [0.,1]')
        print('neuropil_correction_factor set to 0 !')
        neuropil_correction_factor=0.

    #######################################################################


    # Step 1) -> determine the valid ROIs
    valid_roiIndices = (\
            (np.mean(data.rawFluo, axis=1)>\
            roi_to_neuropil_fluo_inclusion_factor*np.mean(data.neuropil, axis=1)))

    # Step 2) ->  performing neuropil correction 
    correctedFluo = data.rawFluo[valid_roiIndices, :]-\
            neuropil_correction_factor*data.neuropil[valid_roiIndices, :]
    
    # Step 3) -> compute the F0 term (~ sliding minimum/percentile)
    correctedFluo0 = compute_sliding_F0(data, correctedFluo,
                                        method=method_for_F0,
                                        percentile=percentile,
                                        sliding_window=sliding_window)

    # Step 4) -> compute the delta F over F quantity: dFoF = (F-F0)/F0
    data.dFoF = (correctedFluo-correctedFluo0)/correctedFluo0


    #######################################################################
    if verbose:
        if np.sum(~valid_roiIndices)>0:
            print('\n  ** %i ROIs were discarded with the positive F0 criterion (%.1f%%) ** \n'\
                  % (np.sum(~valid_roiIndices),
                      100*np.sum(~valid_roiIndices)/correctedFluo.shape[0]))
        else:
            print('\n  ** all ROIs passed the positive F0 criterion ** \n')
            
    # we update the previous quantities
    data.valid_roiIndices = np.arange(data.iscell.sum())[valid_roiIndices]
    data.vNrois= len(data.valid_roiIndices) # number of valid ROIs
    data.rawFluo = data.rawFluo[valid_roiIndices,:]
    data.neuropil = data.neuropil[valid_roiIndices,:]
    data.nROIs = data.vNrois

    if with_correctedFluo_and_F0:
        data.correctedFluo0 = correctedFluo0
        data.correctedFluo = correctedFluo
    
    if verbose:
        print('-> dFoF calculus done !  (calculation took %.1fs)' % (time.time()-tick))

    return None
