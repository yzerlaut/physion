"""
optical physiology: calcium imaging (suite2p output)

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
import time
import numpy as np
from physion.imaging.Calcium import METHOD, NEUROPIL_CORRECTION_FACTOR, PERCENTILE, ROI_TO_NEUROPIL_INCLUSION_FACTOR, ROI_TO_NEUROPIL_INCLUSION_FACTOR_METRIC, T_SLIDING,\
        compute_F0, compute_neuropil_factor
from physion.imaging.dcnv import oasis
from physion.analysis import tools
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

# quantities of shape (nROIs, nTimes), restricted to the valid ROIs by set_valid_roiIndices
OPHYS_QUANTITIES = ['rawFluo', 'neuropil', 'dFoF', 'correctedFluo', 'F0']

class OphysMixin:

    def has_rawFluo(self):
        return self.has_ophys()

    def has_neuropil(self):
        return self.has_ophys()

    def has_dFoF(self):
        return self.has_ophys()

    def has_ophys(self):
        return ('ophys' in self.nwbfile.processing)

    def read_ophys(self):
       
        self.TSeries_folder = self.nwbfile.acquisition[\
                'CaImaging-TimeSeries'].comments.split('**')[-1]

        ### ROI activity ###
        self.Fluorescence = \
                getattr(\
                    getattr(self.nwbfile.processing['ophys'],
                        'data_interfaces')['Fluorescence'],
                            'roi_response_series')['Fluorescence']
        self.Neuropil = \
                getattr(\
                    getattr(self.nwbfile.processing['ophys'],
                        'data_interfaces')['Neuropil'],
                            'roi_response_series')['Neuropil']
        self.CaImaging_dt = (self.Neuropil.timestamps[1]-\
                                    self.Neuropil.timestamps[0])

        ### ROI properties ###
        self.Segmentation = \
                getattr(\
                    getattr(self.nwbfile.processing['ophys'],
                        'data_interfaces')['ImageSegmentation'],
                            'plane_segmentations')['PlaneSegmentation']
        self.pixel_masks_index = self.Segmentation.columns[0].data[:]
        self.pixel_masks = self.Segmentation.columns[1].data[:]

        self.initialize_ROIs()

    def initialize_ROIs(self, 
                        valid_roiIndices=None):

        """
        we read the table properties of the suite2p Segmentation

        we always restart from the original ROIs and only after we apply
                the valid_roiIndices filter
        """

        self.original_nROIs = self.Segmentation.columns[0].data.shape[0]

        # initialize rois properties to default values
        planeID = np.zeros(self.original_nROIs, dtype=int)
        redcell = np.zeros(self.original_nROIs, dtype=bool) 

        # looping over the table properties (0,1 -> rois locs)
        #      for the ROIS to overwrite the defaults:
        for i in range(2, len(self.Segmentation.columns)):
            if self.Segmentation.columns[i].name=='plane':
                planeID = self.Segmentation.columns[i].data[:].astype(int)
            if self.Segmentation.columns[i].name=='redcell':
                redcell = self.Segmentation.columns[i].data[:,0].astype(bool)

        # now we apply the filter if needed:

        if valid_roiIndices is None:
            self.valid_roiIndices = np.arange(self.original_nROIs)
        else:
            self.valid_roiIndices = valid_roiIndices

        self.nROIs = len(self.valid_roiIndices)
        self.planeID = planeID[self.valid_roiIndices]
        self.redcell= redcell[self.valid_roiIndices]


    def set_valid_roiIndices(self,
                             roi_to_neuropil_fluo_inclusion_factor=ROI_TO_NEUROPIL_INCLUSION_FACTOR,
                             roi_to_neuropil_fluo_inclusion_factor_metric=ROI_TO_NEUROPIL_INCLUSION_FACTOR_METRIC,
                             discard_negative_F0=False,
                             discard_negative_neuropil_correction_factor=False,
                             verbose=True):
        """
        restricts the ROIs to the ones passing the selection criteria,
            applied in this order:

        1) [optional] strictly positive baseline: min(F0) > 1
                (needs self.F0)
        2) [optional] strictly positive neuropil correction factor
                (needs self.neuropil_correction_factors, see build_neuropil_correction_factors)
        3) fluorescence above "roi_to_neuropil_fluo_inclusion_factor" times the neuropil,
                for the "mean" or the "std" (after 10-frame smoothing) of the signals
                (None or 0: no criterion)

        the ophys quantities (OPHYS_QUANTITIES) and the ROI properties are restricted
            to the valid ROIs, the selection is relative to the current ROIs
        returns the boolean mask of the kept ROIs (among the current ROIs)
        """

        valid = np.ones(self.nROIs, dtype=bool)

        def apply_criterion(condition, label):
            """ removes the ROIs not satisfying "condition" and reports it """
            removed = valid & ~condition
            valid[:] = valid & condition
            if verbose:
                print('  - %-45s %4i ROIs discarded (%5.1f%%)  -> %4i/%i valid ROIs (%5.1f%%)' % (
                      label, np.sum(removed), 100.*np.sum(removed)/len(valid),
                      np.sum(valid), len(valid), 100.*np.sum(valid)/len(valid)))

        if verbose:
            print('\nselection of the ROIs (n=%i):' % self.nROIs)

        # 1) ROIs with a strictly positive baseline
        if discard_negative_F0:
            if not hasattr(self, 'F0'):
                raise AttributeError('"F0" needed to discard the ROIs with a negative baseline,'+\
                                     ' compute it first')
            apply_criterion(np.min(self.F0, axis=1)>1, 'positive F0 baseline')

        # 2) ROIs with a strictly positive neuropil correction factor
        if discard_negative_neuropil_correction_factor:
            if not hasattr(self, 'neuropil_correction_factors'):
                raise AttributeError('"neuropil_correction_factors" needed to discard the ROIs with'+\
                                     ' a negative factor, see build_neuropil_correction_factors')
            apply_criterion(self.neuropil_correction_factors>0, 'positive neuropil correction factor')

        # 3) ROIs with a fluorescence above the neuropil (inclusion factor)
        if roi_to_neuropil_fluo_inclusion_factor:
            if roi_to_neuropil_fluo_inclusion_factor_metric=='mean':
                fluo = np.mean(self.rawFluo, axis=1)
                neuropil = np.mean(self.neuropil, axis=1)
            elif roi_to_neuropil_fluo_inclusion_factor_metric=='std':
                fluo = np.std(gaussian_filter1d(self.rawFluo, 10, axis=1), axis=1)
                neuropil = np.std(gaussian_filter1d(self.neuropil, 10, axis=1), axis=1)
            else:
                raise ValueError('roi_to_neuropil_fluo_inclusion_factor_metric must be'+\
                                 ' either "mean" or "std"')
            apply_criterion(fluo>roi_to_neuropil_fluo_inclusion_factor*neuropil,
                            'fluo (%s) > %.2f x neuropil' % (roi_to_neuropil_fluo_inclusion_factor_metric,
                                                             roi_to_neuropil_fluo_inclusion_factor))

        # restrict the ROI properties and the ophys quantities to the valid ROIs
        nROIs = self.nROIs
        self.initialize_ROIs(valid_roiIndices=self.valid_roiIndices[valid])
        for quantity in OPHYS_QUANTITIES:
            if hasattr(self, quantity):
                if getattr(self, quantity).shape[0]==len(valid):
                    setattr(self, quantity, getattr(self, quantity)[valid, :])
                else:
                    # computed for another set of ROIs (e.g. a previous selection)
                    delattr(self, quantity)
                    if verbose:
                        print('  [!!] "%s" (previous ROIs) removed, rebuild it' % quantity)
        for factor in ['neuropil_correction_factor', 'neuropil_correction_factors']:
            if np.ndim(getattr(self, factor, None))==1: # one factor per ROI
                setattr(self, factor, getattr(self, factor)[valid])

        if verbose:
            print('  => %i/%i valid ROIs (%.1f%%)\n' % (self.nROIs, nROIs, 100.*self.nROIs/nROIs))

        return valid

    def build_rawFluo(self,
                      specific_time_sampling=None,
                      interpolation='linear',
                      verbose=True):
        """
        same than above for neuropil

        [!!] always built for all ROIs [!!]
                (the valid_roiIndices filter will be applied in build_dFoF)
        """
        self.t_rawFluo = self.Fluorescence.timestamps[:]

        if len(self.t_rawFluo)==self.Fluorescence.data.shape[1]:
            self.rawFluo = np.array(self.Fluorescence.data)
        else:
            # data badly oriented --> transpose in that case
            self.rawFluo = np.array(self.Fluorescence.data).T

        if specific_time_sampling is not None:
            # we first interpolate and resample the data
            self.rawFluo2 = np.zeros((self.rawFluo.shape[0], len(specific_time_sampling)))
            for i in range(self.rawFluo.shape[0]):
                self.rawFluo2[i,:] = tools.resample(self.t_rawFluo,
                                               self.rawFluo[i,:],
                                               specific_time_sampling,
                                               interpolation=interpolation,
                                               verbose=verbose)
            self.rawFluo = self.rawFluo2
            # then we update the timestamps
            self.t_rawFluo= specific_time_sampling


    def build_neuropil(self,
                       specific_time_sampling=None,
                       interpolation='linear',
                       verbose=True):
        """
        we build the neuropil matrix in the form (nROIs, time_samples)
            we need to deal with the fact that matrix orientation 
            was changed because of pynwb complains

        [!!] always built for all ROIs [!!]
                (the valid_roiIndices filter will be applied in build_dFoF)
        """
        self.t_neuropil = self.Neuropil.timestamps[:]

        if len(self.t_neuropil)==self.Neuropil.data.shape[1]:
            self.neuropil = np.array(self.Neuropil.data)[:,:]
        else:
            # data badly oriented --> transpose in that case
            self.neuropil = np.array(self.Neuropil.data).T

        if specific_time_sampling is not None:
            # we first interpolate and resample the data
            self.neuropil2 = np.zeros((self.neuropil.shape[0], len(specific_time_sampling)))
            for i in range(self.neuropil.shape[0]):
                self.neuropil2[i,:] = tools.resample(self.t_neuropil,
                                               self.neuropil[i,:],
                                               specific_time_sampling,
                                               interpolation=interpolation,
                                               verbose=verbose)
            self.neuropil = self.neuropil2
            # then we update the timestamps
            self.t_neuropil = specific_time_sampling

    def build_neuropil_correction_factors(self, **kwargs):
        """
        one neuropil correction factor per ROI: self.neuropil_correction_factors
            (slope of the lower envelope of rawFluo vs neuropil,
             see physion.imaging.Calcium.compute_neuropil_factor for the kwargs)
        """
        self.neuropil_correction_factors = compute_neuropil_factor(self.rawFluo, self.neuropil,
                                                                   **kwargs)

    def build_correctedFluo(self,
                            neuropil_correction_factor=NEUROPIL_CORRECTION_FACTOR,
                            verbose=True):
        """
        self.correctedFluo = rawFluo - neuropil_correction_factor x neuropil

            neuropil_correction_factor: a number in [0,1] (set to 0 otherwise),
                                        or an array with one factor per ROI
            (needs self.rawFluo and self.neuropil)
        """
        factor = np.asarray(neuropil_correction_factor, dtype=float)
        if (factor.ndim==0) and not (0<=factor<=1):
            print('[!!] neuropil_correction_factor has to be in the interval [0.,1]')
            print('neuropil_correction_factor set to 0 !')
            factor = np.asarray(0.)

        # the factor applied (a python float keeps the precision of the data, e.g. float32)
        if factor.ndim==0:
            self.neuropil_correction_factor = float(factor)
            self.correctedFluo = self.rawFluo-self.neuropil_correction_factor*self.neuropil
        else:
            self.neuropil_correction_factor = factor
            self.correctedFluo = self.rawFluo-factor[:, np.newaxis]*self.neuropil

    def build_F0(self,
                 method=METHOD,
                 percentile=PERCENTILE,
                 sliding_window=T_SLIDING):
        """
        self.F0: baseline of the corrected fluorescence
            (see physion.imaging.Calcium.compute_F0 for the methods)
            (needs self.correctedFluo)
        """
        # time interval of correctedFluo (rawFluo, possibly resampled)
        dt = self.t_rawFluo[1]-self.t_rawFluo[0]
        F0 = compute_F0(self.correctedFluo, dt,
                        method=method,
                        percentile=percentile,
                        sliding_window=sliding_window)
        if F0 is None:
            raise ValueError('method "%s" for F0 not recognized' % method)
        self.F0 = F0

    def build_dFoF(self,
                   neuropil_correction_factor=NEUROPIL_CORRECTION_FACTOR,
                   with_computed_neuropil_fact=False,
                   method_for_F0=METHOD,
                   percentile=PERCENTILE,
                   sliding_window=T_SLIDING,
                   roi_to_neuropil_fluo_inclusion_factor=ROI_TO_NEUROPIL_INCLUSION_FACTOR,
                   roi_to_neuropil_fluo_inclusion_factor_metric=ROI_TO_NEUROPIL_INCLUSION_FACTOR_METRIC,
                   smoothing=None,
                   specific_time_sampling=None,
                   interpolation='linear',
                   with_correctedFluo_and_F0=False,
                   verbose=True):
        """
        creates self.dFoF = (correctedFluo-F0)/F0 and self.t_dFoF, from the raw data:

        1) rawFluo and neuropil, of all ROIs (optionally resampled)
                        -> build_rawFluo, build_neuropil
        2) [if with_computed_neuropil_fact] neuropil correction factor of each ROI,
                the mean of the positive ones is applied
                        -> build_neuropil_correction_factors
        3) correctedFluo = rawFluo - neuropil_correction_factor x neuropil
                        -> build_correctedFluo
        4) baseline F0 of correctedFluo   -> build_F0
        5) selection of the ROIs: positive F0, [positive neuropil correction factor],
                fluorescence above the neuropil  -> set_valid_roiIndices
        6) dFoF = (correctedFluo-F0)/F0, [optional] Gaussian smoothing (in frames)

        each step is a method that can also be used on its own

        self.correctedFluo and self.F0 are kept if with_correctedFluo_and_F0
        """
        if verbose:
            tick = time.time()
            print('\ncalculating dF/F with method "%s" [...]' % method_for_F0)

        # 1) all ROIs, from the raw data (removes the previous selection)
        self.initialize_ROIs()
        for quantity in OPHYS_QUANTITIES+['neuropil_correction_factors']:
            if quantity not in ['rawFluo', 'neuropil'] and hasattr(self, quantity):
                delattr(self, quantity)
        self.build_rawFluo(specific_time_sampling=specific_time_sampling,
                           interpolation=interpolation,
                           verbose=verbose)
        self.build_neuropil(specific_time_sampling=specific_time_sampling,
                            interpolation=interpolation,
                            verbose=verbose)
        self.t_dFoF = self.t_rawFluo

        # 2) neuropil correction factor of each ROI
        if with_computed_neuropil_fact:
            self.build_neuropil_correction_factors()
            positive = (self.neuropil_correction_factors>0)
            neuropil_correction_factor = np.mean(self.neuropil_correction_factors[positive])
            if verbose:
                print('neuropil correction factor computed: %.3f' % neuropil_correction_factor)

        # 3) neuropil correction
        self.build_correctedFluo(neuropil_correction_factor=neuropil_correction_factor,
                                 verbose=verbose)

        # 4) baseline
        self.build_F0(method=method_for_F0,
                      percentile=percentile,
                      sliding_window=sliding_window)

        # 5) ROI selection
        self.set_valid_roiIndices(\
                discard_negative_F0=True,
                discard_negative_neuropil_correction_factor=with_computed_neuropil_fact,
                roi_to_neuropil_fluo_inclusion_factor=roi_to_neuropil_fluo_inclusion_factor,
                roi_to_neuropil_fluo_inclusion_factor_metric=\
                        roi_to_neuropil_fluo_inclusion_factor_metric,
                verbose=verbose)

        # 6) delta F over F
        self.dFoF = (self.correctedFluo-self.F0)/self.F0
        if smoothing is not None:
            self.dFoF = gaussian_filter1d(self.dFoF, smoothing, axis=1)

        if not with_correctedFluo_and_F0:
            del self.correctedFluo, self.F0

        if verbose:
            print('-> dFoF calculus done !  (calculation took %.1fs)' % (time.time()-tick))

    def build_Deconvolved(self, Tau=1.3, 
                          quantity='correctedFluo',
                          verbose=False):
        """
        use the oasis library to deconvolve the fluorescence signals of choice (default: dFoF)
        """

        if hasattr(self, quantity): 

            self.quantity_Deconvolved = quantity
            self.t_Deconvolved = self.t_dFoF

            # time interval of the signal (possibly resampled)
            dt = self.t_Deconvolved[1]-self.t_Deconvolved[0]
            self.Deconvolved = \
                oasis(getattr(self, quantity),
                        getattr(self, quantity).shape[0], # batch size
                            Tau, 1./dt)

        else: 
            print('\n deconvolution not possible \n --> ' + quantity + ' does not exist')
            print("build your signal before deconvolving using 'setattr(data, name of your signal, values of your signal)'")
