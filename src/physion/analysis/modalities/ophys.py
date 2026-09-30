"""
optical physiology: calcium imaging (suite2p output)

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
import numpy as np
from physion.imaging.Calcium import METHOD, NEUROPIL_CORRECTION_FACTOR, PERCENTILE, ROI_TO_NEUROPIL_INCLUSION_FACTOR, ROI_TO_NEUROPIL_INCLUSION_FACTOR_METRIC, T_SLIDING, compute_dFoF
from physion.imaging.dcnv import oasis
from physion.analysis import tools


class OphysMixin:

    def has_rawFluo(self):
        return self.has_ophys()

    def has_neuropil(self):
        return self.has_ophys()

    def has_dFoF(self):
        return self.has_ophys()

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

    def build_dFoF(self,
                   roiIndex=None, 
                   roi_to_neuropil_fluo_inclusion_factor=ROI_TO_NEUROPIL_INCLUSION_FACTOR,
                   neuropil_correction_factor=NEUROPIL_CORRECTION_FACTOR,
                   method_for_F0=METHOD,
                   percentile=PERCENTILE,
                   sliding_window=T_SLIDING,
                   with_correctedFluo_and_F0=False,
                   specific_time_sampling=None,
                   smoothing=None,
                   interpolation='linear',
                   with_computed_neuropil_fact=False,
                   roi_to_neuropil_fluo_inclusion_factor_metric=ROI_TO_NEUROPIL_INCLUSION_FACTOR_METRIC,
                   verbose=True):
        """
        creates self.dFoF, self.t_dFoF

        [!!] we always rebuild the rawFluo and neuropil 
                to remove the potential valid_roiIndices previous filters
        """

        self.build_rawFluo(specific_time_sampling=specific_time_sampling,
                           interpolation=interpolation,
                           verbose=verbose)
        self.build_neuropil(specific_time_sampling=specific_time_sampling,
                            interpolation=interpolation,
                            verbose=verbose)
        self.t_dFoF = self.t_rawFluo

        return compute_dFoF(self,
                            roi_to_neuropil_fluo_inclusion_factor=\
                                    roi_to_neuropil_fluo_inclusion_factor,
                            neuropil_correction_factor=\
                                    neuropil_correction_factor,
                            method_for_F0=method_for_F0,
                            percentile=percentile,
                            sliding_window=sliding_window,
                            with_correctedFluo_and_F0=\
                                    with_correctedFluo_and_F0,
                            smoothing=smoothing,
                            with_computed_neuropil_fact=with_computed_neuropil_fact,
                            roi_to_neuropil_fluo_inclusion_factor_metric=\
                                    roi_to_neuropil_fluo_inclusion_factor_metric,
                            verbose=verbose)

    def build_Zscore_dFoF(self, verbose=True):
        """
        [!!] do not deal with specific time sampling [!!] 
        """

        if not hasattr(self, 'dFoF'):
            self.build_dFoF(verbose=verbose)

        setattr(self, 'Zscore_dFoF', 
            (self.dFoF-self.dFoF.mean(axis=0).reshape(1, self.dFoF.shape[1]))/self.dFoF.std(axis=0).reshape(1, self.dFoF.shape[1]))

    def build_Deconvolved(self, Tau=1.3, 
                          quantity='dFoF',
                          verbose=False):
        """
        use the oasis library to deconvolve the fluorescence signals of choice (default: dFoF)
        """

        if hasattr(self, quantity): 

            self.t_Deconvolved = self.t_dFoF

            # for backward compatibilty
            fsignal = getattr(self, quantity)
            deconv = oasis(fsignal,
                            fsignal.shape[0], # batch size
                              Tau, 1./self.CaImaging_dt)
            setattr(self, 'Deconvolved_' + quantity,
                    deconv)

            self.Deconvolved = deconv

        else: 
            print('\n deconvolution not possible \n --> ' + quantity + ' does not exist')
            print("build your signal before deconvolving using 'setattr(data, name of your signal, values of your signal)'")

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
        if not hasattr(self, 't_neuropil'):
            self.t_neuropil = self.Neuropil.timestamps[:]

        if len(self.t_neuropil)==self.Neuropil.data.shape[1]:
            self.neuropil = np.array(self.Neuropil.data)[:,:]
        else:
            # data badly oriented --> transpose in that case
            self.neuropil = np.array(self.Neuropil.data).T

        if specific_time_sampling is not None:
            # we first interpolate and resample the data
            self.neuropil2 = np.zeros((self.nROIs, len(specific_time_sampling)))
            for i in range(self.nROIs):
                self.neuropil2[i,:] = tools.resample(self.t_neuropil,
                                               self.neuropil[i,:],
                                               specific_time_sampling,
                                               interpolation=interpolation,
                                               verbose=verbose)
            self.neuropil = self.neuropil2
            # then we update the timestamps
            self.t_neuropil = specific_time_sampling

    def build_rawFluo(self,
                      roiIndex=None, roiIndices='all',
                      specific_time_sampling=None,
                      interpolation='linear',
                      verbose=True):
        """
        same than above for neuropil

        [!!] always built for all ROIs [!!]
                (the valid_roiIndices filter will be applied in build_dFoF)
        """
        if not hasattr(self, 't_rawFluo'):
            self.t_rawFluo = self.Fluorescence.timestamps[:]

        if len(self.t_rawFluo)==self.Fluorescence.data.shape[1]:
            self.rawFluo = np.array(self.Fluorescence.data)
        else:
            # data badly oriented --> transpose in that case
            self.rawFluo = np.array(self.Fluorescence.data).T

        if specific_time_sampling is not None:
            # we first interpolate and resample the data
            self.rawFluo2 = np.zeros((self.nROIs, len(specific_time_sampling)))
            for i in range(self.nROIs):
                self.rawFluo2[i,:] = tools.resample(self.t_rawFluo,
                                               self.rawFluo[i,:],
                                               specific_time_sampling,
                                               interpolation=interpolation,
                                               verbose=verbose)
            self.rawFluo = self.rawFluo2
            # then we update the timestamps
            self.t_rawFluo= specific_time_sampling
