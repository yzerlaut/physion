import time, ast, sys, pathlib, os
import pynwb # NWB python API
import numpy as np
from scipy.interpolate import interp1d

from physion.utils.files import get_files_with_extension
from physion.visual_stim.build import build_stim
from physion.analysis import tools
from physion.imaging.Calcium import compute_dFoF,\
        ROI_TO_NEUROPIL_INCLUSION_FACTOR, METHOD,\
        T_SLIDING, PERCENTILE, NEUROPIL_CORRECTION_FACTOR, ROI_TO_NEUROPIL_INCLUSION_FACTOR_METRIC
from physion.imaging.dcnv import oasis

MODALITIES = [\
        'photodiode',
        'visual_stim',
        'pupil',
        'facemotion',
        'running',
        'opto',
        'rawFluo',
        'neuropil',
        'dFoF',
        'spikes',
        'spikeWaveforms',
        'LFP',
        'MUA'
    ]

class Data:
    
    """
    a basic class to read NWB
    this class if thought to be the parent for specific applications

    attributes:

    # visual stimulation
    - data.photodiode    # photodiode trace
    - data.visual_stim   # object: physion.visual_stim.main.VisualStim

    # behavioral monitoring
    - data.pupil         # pupil diameter trace
    - data.facemotion    # motion energy in whisking pad over time 
    - data.running       # running speed time trace

    # ophys
    - data.rawFluo       # raw fluorescence of single ROIs
    - data.neuropil      # neuropil fluorescence associated to single ROIs
    - data.dFoF          # delta F over F trace

    # ephys
    - data.spikes       # single unit spike trains
    - data.spikeWaveforms # single unit spike waveforms
    - data.LFP          # Local Field Potential 
    - data.MUA          # Multi-Unit Activity

    # others/common
    data.metadata       # dictionary of metadata
    data.df_name        # formatted name with protocol

    """

    def __init__(self, filename,
                 with_tlim=True,
                 metadata_only=False,
                 with_visual_stim=False,
                 verbose=False):

        self.filename = filename.split(os.path.sep)[-1]
        self.tlim, self.visual_stim, self.nwbfile = None, None, None
        self.metadata, self.df_name = None, ''
        
        if verbose:
            t0 = time.time()

        if verbose:
            print('starting reading [...]')

        self.io = pynwb.NWBHDF5IO(filename, 'r')
        self.nwbfile = self.io.read()

        self.read_metadata()
        if verbose:
            print(' [ok] -> metadata loaded')
            print(self.metadata)

        if with_tlim:
            self.read_tlim()
            if verbose:
                print(' [ok] -> tlim:', self.tlim)

        if not metadata_only:
            self.read_data()
            if verbose:
                print(' [ok] -> modality-specific metadata loaded ')

        if with_visual_stim:
            self.build_visual_stim(verbose=verbose)
            if verbose:
                print(' [ok] -> visual stim loaded')

        if metadata_only:
            self.close()
            
        if verbose:
            print('NWB-file reading time: %.1fms' % (1e3*(time.time()-t0)))


    def read_metadata(self):
        
        self.df_name = self.nwbfile.session_start_time.strftime(\
                                    "%Y/%m/%d -- %H:%M:%S")+\
                        ' -- '+self.nwbfile.experiment_description
        
        self.metadata = ast.literal_eval(\
            # self.nwbfile.session_description # DOESN'T WORK ANYMORE, tuple instead of string ??? need to replace with below...
            self.nwbfile.session_description.replace('("', '').replace('",)','')
        )

        space = '        '
        self.description = '\n - Subject: %s %s \n' % (space,
                                        self.nwbfile.subject.subject_id)

        if 'protocol' not in self.metadata.keys():
            self.metadata['protocol'] = self.nwbfile.experiment_description

        if self.metadata['protocol']=='None':
            self.description += '\n - Spont. Act. (no visual stim.)\n'
        else:
            self.description += '\n - Visual-Stim: \n %s' % space


        if self.nwbfile.protocol is not None:
            self.metadata |= ast.literal_eval(self.nwbfile.protocol)

        # deal with multi-protocols
        if ('Presentation' in self.metadata) and\
                (self.metadata['Presentation']=='multiprotocol'):
            self.protocols, ii = [], 1
            while ('Protocol-%i' % ii) in self.metadata:
                self.protocols.append(self.metadata['Protocol-%i' % ii].split('/')[-1].replace('.json','').replace('-many',''))
                # self.description += '- %s \n' % self.protocols[ii-1]
                self.description += '%s / ' % self.protocols[ii-1]
                ii+=1
                if ii%3==1:
                    self.description += '\n %s' % space
        else:
            self.protocols = [self.metadata['protocol']]
            if self.metadata['protocol']!='None':
                self.description += '- %s \n' % self.metadata['protocol']

 
        self.protocols = np.array(self.protocols, dtype=str)
        self.metadata['protocols'] = self.protocols

        if 'time_start_realigned' in self.nwbfile.stimulus.keys():
            self.description += '\n        =>  completed N=%i/%i episodes  \n' %(self.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                               self.nwbfile.stimulus['time_start'].data.shape[0])
                
        self.description += '\n - Intervention: %s %s\n' % (space, self.metadata['intervention'] if 'intervention' in self.metadata else 'None')

        self.description += '\n - Notes: %s %s\n' % (space, self.nwbfile.notes)

        if hasattr(self.nwbfile.subject, 'age') and self.nwbfile.subject.age!=None:
            self.age = int(str(self.nwbfile.subject.age).replace('P','').replace('D',''))
        else:
            self.age = -1

    
    def read_tlim(self):
        
        self.tlim, safety_counter = None, 0
        
        while (self.tlim is None) and (safety_counter<20):
            for key in self.nwbfile.acquisition:
                try:
                    self.tlim = [self.nwbfile.acquisition[key].starting_time,
                                 self.nwbfile.acquisition[key].starting_time+\
                                 (self.nwbfile.acquisition[key].data.shape[0]-1)/self.nwbfile.acquisition[key].rate]
                except BaseException as be:
                    safety_counter += 1
                try:
                    self.tlim = [self.nwbfile.acquisition[key].timestamps[0],
                                 self.nwbfile.acquisition[key].timestamps[-1]]
                except BaseException as be:
                    safety_counter += 1

        if self.tlim is None:
            self.tlim = [0, 60*60] # 1h by default (~ upper limit) 


    def read_data(self):
        """
        only reads metadata of modalitites...
            they still need to be built afterwards
        """

        # ophys data
        if self.has_ophys():
            self.read_ophys()
        else:
            for key in ['Segmentation', 'Fluorescence', 'redcell', 'plane',
                        'valid_roiIndices', 'neuropil']:
                setattr(self, key, None)

        # behavioral monitoring
        if self.has_pupil():
            self.read_pupil()
        if self.has_facemotion():
            self.read_facemotion()

    #########################################################
    #       CALCIUM IMAGING DATA (from suite2p output)      #
    #########################################################

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
                

    ######################
    #    LOCOMOTION
    ######################    
    def has_running(self):
        return ('Running-Speed' in self.nwbfile.acquisition)
    
    def build_running(self,
                      specific_time_sampling=None,
                      interpolation='linear',
                      verbose=False, 
                      absolute=True):
        """
        Build running speed from NWB acquisition.
        
        Parameters
        ----------
        specific_time_sampling : array-like or None - If provided, resample running speed to these time points.
        interpolation : str - Interpolation method for resampling ('linear', 'nearest', etc.).
        verbose : bool - If True, print progress messages.
        absolute : bool - If True, running speed values are converted to absolute values.

        Returns
        -------
        running_resampled : np.ndarray - Resampled running speed if specific_time_sampling is provided, otherwise None.
        """
        if self.has_running():
            if absolute:
                self.running = np.abs(self.nwbfile.acquisition['Running-Speed'].data[:, 0])
            else : 
                self.running = self.nwbfile.acquisition['Running-Speed'].data[:, 0]

            self.t_running = tools.build_timestamps(\
                        self.nwbfile.acquisition, 'Running-Speed')

            if verbose:
                print(' [ok] --> "running" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_running,
                                    self.running,
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)

        else:
            print(' %s --> "running" not available ...' % self.df_name)

    
    ######################
    #       PUPIL 
    ######################        

    def has_pupil(self):
        return ('Pupil' in self.nwbfile.processing)
    def has_gaze(self):
        return self.has_pupil()

    def read_pupil(self):
        """
        read metadata
        """

        pd = str(self.nwbfile.processing['Pupil'].description)

        # extract pupil scale
        if len(pd.split('pix_to_mm='))>1:
            self.FaceCamera_mm_to_pix = int(1./float(pd.split('pix_to_mm=')[-1]))
        else:
            self.FaceCamera_mm_to_pix = 1

        # extract pupil ROI
        try:
            self.pupil_ROI = {}
            for key, val in zip(\
                ['xmin','xmax','ymin','ymax'],
                pd.split('pupil ROI: (xmin,xmax,ymin,ymax)=(')[1].split(')')[0].split(',')):
                self.pupil_ROI[key] = int(val)
        except BaseException as be:
            self.pupil_ROI = None


    def build_pupil(self,
                    specific_time_sampling=None,
                    interpolation='linear',
                    verbose=False):
        """
        build pupil diameter trace, i.e. twice the maximum of the ellipse radius at each time point
        """
        if self.has_pupil():

            self.t_pupil = self.nwbfile.processing['Pupil'].data_interfaces['cx'].timestamps
            self.pupil =  2*np.max([self.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:,0],
                                             self.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:,0]], axis=0)

            if verbose:
                print(' [ok] --> "pupil" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_pupil, self.pupil,
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

        else:
            print(' %s --> pupil diameter not available ...' % self.df_name)

    def build_gaze(self,
                            specific_time_sampling=None,
                            interpolation='linear',
                            verbose=False):
        """
        build gaze movement 

        build distance from mean (x,y) position of pupil
        """
        if self.has_pupil():
            self.t_gaze = self.nwbfile.processing['Pupil'].data_interfaces['cx'].timestamps
            cx = self.nwbfile.processing['Pupil'].data_interfaces['cx'].data[:,0]
            cy = self.nwbfile.processing['Pupil'].data_interfaces['cy'].data[:,0]
            self.gaze = np.sqrt((cx-np.mean(cx))**2+(cy-np.mean(cy))**2)

            if specific_time_sampling is not None:
                return tools.resample(self.t_gaze, self.gaze, 
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

            if verbose:
                print(' [ok] --> "gaze" built successfully ')

        else:
            print(' %s --> gaze movement not available ...' % self.df_name)
        

    #########################
    #       FACEMOTION  
    #########################      
    
    def has_facemotion(self):
        return ('FaceMotion' in self.nwbfile.processing)

    def read_facemotion(self):
        
        try:
            fd = str(self.nwbfile.processing['FaceMotion'].description)
            self.FaceMotion_ROI = [int(i) for i in fd.split('y0,dy)=(')[1].split(')')[0].split(',')]
        except BaseException as be:
            self.FaceMotion_ROI = None


    def build_facemotion(self,
                         specific_time_sampling=None,
                         interpolation='linear',
                         verbose=False):
        """
        build facemotion
        """

        if self.has_facemotion():

            self.t_facemotion = self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].timestamps
            self.facemotion =  self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].data[:,0]

            if verbose:
                print(' [ok] --> "facemotion" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_facemotion, self.facemotion, 
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

        else:
            print(' %s --> "facemotion" not available ...' % self.df_name)

    #############################
    #       Optogenetics        #
    #############################

    def has_opto(self):
        return ('OptogeneticSeries' in self.nwbfile.stimulus)

    def build_opto(self,
        specific_time_sampling=None,
        interpolation='linear',
        verbose=True):

        if self.has_opto():

            self.opto = self.nwbfile.stimulus['OptogeneticSeries'].data[:]

            self.t_opto = tools.build_timestamps(\
                        self.nwbfile.stimulus, 'OptogeneticSeries')

            if verbose:
                print(' [ok] --> "running" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_opto,
                                    self.opto,
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)
        else:
            print(' %s --> "opto" not available ...' % self.df_name)



    #############################
    #       Electrophysiology   #
    #############################

    def read_ephys(self):
       
        self.NPX_folder = \
            data.nwbfile.devices['Neuropixels OneBox'].description.split('**')[-1]

    def has_LFP(self):
        return ('LFP' in self.nwbfile.processing)

    def build_LFP(self,
                    specific_time_sampling=None,
                    interpolation='linear',
                    verbose=True):

        if self.has_LFP():

            # we transpose to have a matrix of shape (channels, timestamps)
            self.LFP = np.transpose(\
                self.nwbfile.processing['LFP'].data_interfaces['LFP'].data[:])
            self.t_LFP = self.nwbfile.processing['LFP'].data_interfaces['LFP'].timestamps[:]

            if verbose:
                print(' [ok] --> "LFP" built successfully ')

            if specific_time_sampling is not None:
                return np.array([\
                    tools.resample(self.t_LFP,
                                    self.LFP[i,:],
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)\
                                    for i in range(self.LFP.shape[0])])
        else:
            print(' %s --> "LFP" not available ...' % self.df_name)
            

    def has_MUA(self):
        return ('MUA' in self.nwbfile.processing)

    def build_MUA(self,
                    specific_time_sampling=None,
                    interpolation='linear',
                    verbose=True):

        if self.has_MUA():

            # we transpose to have a matrix of shape (channels, timestamps)
            self.MUA = np.transpose(\
                self.nwbfile.processing['MUA'].data_interfaces['MUA'].data[:])
            self.t_MUA = self.nwbfile.processing['MUA'].data_interfaces['MUA'].timestamps[:]

            if verbose:
                print(' [ok] --> "MUA" built successfully ')

            if specific_time_sampling is not None:
                return np.array([\
                    tools.resample(self.t_MUA,
                                    self.MUA[i,:],
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)\
                                    for i in range(self.MUA.shape[0])])
        else:
            print(' %s --> "MUA" not available ...' % self.df_name)

    def has_spikes(self):
        return getattr(self.nwbfile, 'units')!=None

    def build_spikes(self,
            specific_time_sampling=None,
            dt=1e-3,
            interpolation='linear',
            verbose=True):
        """
        single-unit Spikes

        builds a matrix (units, times) of boolean values
            True -> means spike at that time for that unit


        by default: dt=1ms
        """
        if self.has_spikes():

            n = int((self.tlim[1]-self.tlim[0])/dt)
            self.t_spikes = np.arange(n)*dt
            self.spikes = np.zeros(\
                (len(self.nwbfile.units), n), dtype=bool)
            
            for i, unit in enumerate(self.nwbfile.units):
                for s in unit.spike_times.values[:][0]:
                    if int(s/dt)<n:
                        self.spikes[i, int(s/dt)] = True

            if verbose:
                print(' [ok] --> "spikes" built successfully ')

            if specific_time_sampling is not None:
                return np.array([\
                    tools.resample(self.t_spikes,
                                    self.spikes[i,:],
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)\
                                    for i in range(self.spikes.shape[0])])

        else:
            print(' %s --> "spikes" not available ...' % self.df_name)


    def has_spikeWaveforms(self):
        return ('Spiking' in self.nwbfile.processing) and\
            ('single-unit Waveforms' in self.nwbfile.processing['Spiking'].data_interfaces)

    def build_spikeWaveforms(self,
                             verbose=False):
        """ 
        load the spike template waveforms 
        """
        if self.has_spikeWaveforms():

            k1, k2 = 'Spiking', 'single-unit Waveforms'
            self.t_spikeWaveforms = self.nwbfile.processing[k1].data_interfaces[k2].times[:]
            self.spikeWaveforms = self.nwbfile.processing[k1].data_interfaces[k2].features[:]

            if verbose:
                print(' [ok] --> "spikeWaveforms" built successfully ')
        else:
            print(' %s --> "spikeWaveforms" not available ...' % self.df_name)




    def build_muEvents(self,
            specific_time_sampling=None,
            dt=1e-3,
            interpolation='linear',
            verbose=True):
        """
        multi-unit peak detection Events

        builds a matrix (channels, times) of integer values
            True -> means spike at that time for that unit


        by default: dt=1ms
        """
        n = int((self.tlim[1]-self.tlim[0])/dt)
        self.t_muEvents= np.arange(n)*dt
        self.muEvents= np.zeros(\
            (len(self.nwbfile.electrodes), n), dtype=np.uint8)
        
        channels = self.nwbfile.processing['Spiking'].data_interfaces['multi-unit Events'].data[:]
        times = self.nwbfile.processing['Spiking'].data_interfaces['multi-unit Events'].data[:]
        for chan in np.unique(channels):
            channel_cond = channels==chan
            for s in times[channel_cond]:
                if int(s/dt)<n:
                    self.muEvents[chan, int(s/dt)] += 1

        if specific_time_sampling is not None:
            return np.array([\
                tools.resample(self.t_muEvents,
                                self.muEvents[i,:],
                                specific_time_sampling,
                                interpolation=interpolation,
                                verbose=verbose)\
                                for i in range(self.muEvents.shape[0])])

    #############################
    #       Calcium Imaging     #
    #############################

        
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

    def build_Deconvolved(self, Tau=1.3):
        """
        use the oasis library to deconvolve the dFoF signals
        """
        if not hasattr(self, 'dFoF'):
            print('\n deconvolution not possible \n --> need to build_dFoF(**options) first !! ')
        else:
            self.t_Deconvolved = self.t_dFoF
            setattr(self, 'Deconvolved',
                    oasis(self.dFoF, 
                          self.dFoF.shape[0], # batch size
                              Tau, 1./self.CaImaging_dt))


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


    ################################################
    #       episodes and visual stim protocols     #
    ################################################

    def has_photodiode(self):
        return ('Photodiode-Signal' in self.nwbfile.acquisition)

    def build_photodiode(self,
                      specific_time_sampling=None,
                      interpolation='linear',
                      verbose=False):

        if self.has_photodiode():

            self.photodiode = self.nwbfile.acquisition[\
                                    'Photodiode-Signal'].data[:, 0]

            self.t_photodiode = tools.build_timestamps(\
                    self.nwbfile.acquisition, 'Photodiode-Signal')

            if verbose:
                print(' [ok] --> "photodiode" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_photodiode,
                                    self.photodiode,
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)
            else:
                return None
        else:
            print(' %s --> photodiode not available ...' % self.df_name)
            return None



    def has_visual_stim(self):
        return ('time_start_realigned' in self.nwbfile.stimulus)

    def build_visual_stim(self, 
                         verbose=False, 
                         force_degree=False):
        """
        Builds an initial visual stim  - well built?
        Overwrites it by: 
        Looping for each episode: 
            Looks for keys that are both in the experiment keys and the stimulus keys and 
            for each one : 
                the value from the NWB file is stored in the good place in self.visual_stim.experiment

        if force_degree=True : forces degrees when re-initializing from data (for plots in degrees)

        """

        if self.has_visual_stim():

            self.metadata['verbose'] = verbose
            if force_degree:
                self.metadata['units'] = 'deg'

            # build an initial visual_stim 
            self.visual_stim = build_stim(protocol=self.metadata)
            
            # then force to what was really shown (NWB file)
            for i in range(self.nwbfile.stimulus['time_start_realigned'].num_samples):
                for key in self.visual_stim.experiment: 
                    if key in self.nwbfile.stimulus:
                        self.visual_stim.experiment[key][i]=\
                            self.nwbfile.stimulus[key].data[i,0]
                    
            if force_degree and\
                hasattr(self.visual_stim, 'STIM'):
                for s in self.visual_stim.STIM:
                    s.set_angle_meshgrid(force_degree=True)

            if verbose:
                print(' [ok] --> "visual_stim" built successfully ')
        else:
            print(' %s --> visual stim not available ...' % self.df_name)
        
    def get_protocol_id(self, protocol_name):
        cond = np.argwhere(self.protocols==protocol_name).flatten()
        if len(cond)==1:
            return cond[0]
        else:
            print(' [!!] protocol "%s" not found in data with protocols:' % protocol_name)
            print(self.protocols)
            return None

    
    def get_protocol_cond(self, protocol_id, protocol_name=None):
        """
        ## a recording can have multiple protocols inside
        -> find the condition of a given protocol ID

        'None' to have them all 
        """

        if (protocol_name is not None) and (('protocol_id' in self.nwbfile.stimulus) and\
                (len(np.unique(self.nwbfile.stimulus['protocol_id'].data[:,0]))>1)):
            protocol_id = self.get_protocol_id(protocol_name)
            Pcond = (self.nwbfile.stimulus['protocol_id'].data[:,0]==protocol_id)

        elif (protocol_id is not None) and (('protocol_id' in self.nwbfile.stimulus) and\
                (len(np.unique(self.nwbfile.stimulus['protocol_id'].data[:,0]))>1)):
            Pcond = (self.nwbfile.stimulus['protocol_id'].data[:,0]==protocol_id)

        else:
            # print('no protocol ID')
            Pcond = np.ones(self.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)
             
        # limiting to available episodes
        Pcond[np.arange(len(Pcond))>=self.nwbfile.stimulus['time_start_realigned'].num_samples] = False

        return Pcond
        
    
    def get_stimulus_conditions(self, X, K, protocol_id):
        """
        find the episodes where the keys "K" have the values "X"
        """
        Pcond = self.get_protocol_cond(protocol_id)
        
        if len(K)>0:
            CONDS = []
            XK = np.meshgrid(*X)
            for i in range(len(XK[0].flatten())): # looping over joint conditions
                cond = np.ones(np.sum(Pcond), dtype=bool)
                for k, xk in zip(K, XK):
                    cond = cond & (self.nwbfile.stimulus[k].data[Pcond,0]==xk.flatten()[i])
                CONDS.append(cond)
            return CONDS
        else:
            return [np.ones(np.sum(Pcond), dtype=bool)]


    def find_episode_from_time(self, time):
        """
        returns episode number
                -1 if prestim, interstim, or poststim
        """
        if 'time_start_realigned' in self.nwbfile.stimulus:
            start_key, stop_key = 'time_start_realigned', 'time_stop_realigned'
        else:
            start_key, stop_key = 'time_start', 'time_stop'

        cond = (time>=self.nwbfile.stimulus[start_key].data[:,0]) & (time<=self.nwbfile.stimulus[stop_key].data[:,0])

        if np.sum(cond)>0:
            return np.arange(self.nwbfile.stimulus[start_key].num_samples)[cond][0]
        else:
            return -1

        
    ###########################
    #       other methods     #
    ###########################
    
    def available_modalities(self,
                             verbose=False):
        self.modalities = []
        for key in MODALITIES:
            if getattr(self, 'has_%s' % key)():
                self.modalities.append(key)
                if verbose:
                    print(' --> available: "%s" ' % key)
        return self.modalities

    def build_available_modalities(self,
                                   verbose=True):
        for key in self.available_modalities(verbose=verbose):
            getattr(self, 'build_%s' % key)(verbose=verbose)

    def close(self):
        self.io.close()

        
def scan_folder_for_NWBfiles(folder, 
                             for_protocol=None,
                             for_protocols=[],
                             sorted_by='filename',
                             Nmax=1000000,
                             exclude_intrinsic_imaging_files=True,
                             verbose=True):
    """
    scan folders for protocols and returns a list of datafiles

    by default: excludes the intrinsic imaging files
    """
    if verbose:
        print('inspecting the folder "%s" [...]' % folder)
        t0 = time.time()

    if (for_protocol is not None) and (len(for_protocols)==0):
        for_protocols = [for_protocol]

    FILES0 = get_files_with_extension(folder,
                    extension='.nwb', recursive=True)
    
    if exclude_intrinsic_imaging_files:
        FILES0 = [f for f in FILES0 if (('left-' not in f) and\
                                      ('down-' not in f) and\
                                      ('right-' not in f) and\
                                      ('up-' not in f))]

    DATES = np.array([f.split(os.path.sep)[-1].split('-')[0] for f in FILES0])
    FILES, SUBJECTS, PROTOCOLS, PROTOCOL_IDS, AGES = [], [], [], [], []

    for f in FILES0[:Nmax]:

        try:
            data = Data(f, metadata_only=True, verbose=False)

            if len(for_protocols)>0:

                # we look for specific protocols
                iProtocols, Protocols = [], []
                for protocol in for_protocols:
                    iP = np.flatnonzero(data.protocols==protocol)
                    if len(iP)==1:
                        iProtocols.append(iP[0])
                        Protocols.append(data.protocols[iP[0]])

                if len(Protocols)>0:
                    # if it has at least one protocol, we include it
                    FILES.append(f)
                    PROTOCOLS.append(Protocols)
                    PROTOCOL_IDS.append(iProtocols)
                    SUBJECTS.append(data.nwbfile.subject.subject_id)
                    AGES.append(data.age)

            else:

                # we include with all protocols
                FILES.append(f)
                PROTOCOLS.append(data.protocols)
                PROTOCOL_IDS.append(range(len(data.protocols)))
                SUBJECTS.append(data.nwbfile.subject.subject_id)
                AGES.append(data.age)

        except BaseException as be:
            SUBJECTS.append('N/A')
            if verbose:
                print(be)
                print('\n [!!] Pb with "%s" \n' % f)
        
    if verbose:
        print(' -> found n=%i datafiles (in %.1fs) ' % (len(FILES),
                                                        (time.time()-t0)))

    # sorted by filename

    if sorted_by=='filename':
        isorted = np.argsort(FILES)
    elif sorted_by=='subject':
        isorted = np.argsort(SUBJECTS)
    elif sorted_by=='date':
        isorted = np.argsort(DATES)
    elif sorted_by=='age':
        isorted = np.argsort(AGES)
    else:
        print(' "%s" no recognized , --> sorted by filename by default ! ' % sorted_by)
        isorted = np.argsort(FILES)

    return {'files':np.array(FILES)[isorted], 
            'dates':np.array(DATES)[isorted],
            'subjects':np.array(SUBJECTS)[isorted],
            'ages':np.array(AGES)[isorted],
            'protocol_ids':[PROTOCOL_IDS[i] for i in isorted],
            'protocols':[PROTOCOLS[i] for i in isorted]}


if __name__=='__main__':

    if '.nwb' in sys.argv[-1]:
        import pprint
        data = Data(sys.argv[-1], verbose=True)
        pprint.pprint(data.metadata)
        print()
        for key in data.available_modalities(True):
            # building modalities:
            getattr(data, 'build_%s' % key)(verbose=True)
    else:
        datafolder = sys.argv[-1]
        DATASET = \
            scan_folder_for_NWBfiles(datafolder)
        print(DATASET)