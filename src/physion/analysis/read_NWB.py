import time, ast, sys, pathlib, os
import pynwb # NWB python API
import numpy as np
from scipy.interpolate import interp1d

from physion.utils.files import get_files_with_extension
from physion.visual_stim.build import build_stim
from physion.analysis import tools
from physion.imaging.Calcium import compute_dFoF,\
        ROI_TO_NEUROPIL_INCLUSION_FACTOR, METHOD,\
        T_SLIDING, PERCENTILE, NEUROPIL_CORRECTION_FACTOR
from physion.imaging.dcnv import oasis

class Data:
    
    """
    a basic class to read NWB
    this class if thought to be the parent for specific applications
    """
    
    def __init__(self, filename,
                 with_tlim=True,
                 metadata_only=False,
                 with_visual_stim=False,
                 verbose=True):

        self.filename = filename.split(os.path.sep)[-1]
        self.tlim, self.visual_stim, self.nwbfile = None, None, None
        self.metadata, self.df_name = None, ''
        
        if verbose:
            t0 = time.time()

        # try:
        self.io = pynwb.NWBHDF5IO(filename, 'r')
        self.nwbfile = self.io.read()

        self.read_metadata()

        if with_tlim:
            self.read_tlim()

        if not metadata_only:
            self.read_data()

        if with_visual_stim:
            self.init_visual_stim(verbose=verbose)

        if metadata_only:
            self.close()
            
        # except BaseException as be:
        #     print('-----------------------------------------')
        #     print(be)
        #     print('-----------------------------------------')
        #     print(' /!\ Pb with datafile: "%s"' % filename)
        #     print('-----------------------------------------')
        #     print('')
            
        if verbose:
            print('NWB-file reading time: %.1fms' % (1e3*(time.time()-t0)))


    def read_metadata(self):
        
        self.df_name = self.nwbfile.session_start_time.strftime(\
                                    "%Y/%m/%d -- %H:%M:%S")+\
                        ' ---- '+self.nwbfile.experiment_description
        
        self.metadata = ast.literal_eval(\
                self.nwbfile.session_description)

        space = '        '
        self.description = '\n - Subject: %s %s \n' % (space,
                                        self.metadata['subject_ID'])

        if self.metadata['protocol']=='None':
            self.description += '\n - Spont. Act. (no visual stim.)\n'
        else:
            self.description += '\n - Visual-Stim: \n %s' % space

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

        self.description += '\n - Notes: %s %s\n' % (space, self.metadata['notes'])

        # FIND A BETTER WAY TO DESCRIBE
        # if self.metadata['protocol']!='multiprotocols':
        #     self.keys = []
        #     for key in self.nwbfile.stimulus.keys():
        #         if key not in ['index', 'time_start', 'time_start_realigned',
        #                        'time_stop', 'time_stop_realigned', 'visual-stimuli', 'frame_run_type']:
        #             if len(np.unique(self.nwbfile.stimulus[key].data[:]))>1:
        #                 s = '-*  N-%s = %i' % (key,len(np.unique(self.nwbfile.stimulus[key].data[:])))
        #                 self.description += s+(35-len(s))*' '+'[%.1f, %.1f]\n' % (np.min(self.nwbfile.stimulus[key].data[:]),
        #                                                                         np.max(self.nwbfile.stimulus[key].data[:]))
        #                 self.keys.append(key)
        #             else:
        #                 self.description += '- %s=%.1f\n' % (key, np.unique(self.nwbfile.stimulus[key].data[:]))
                    
        
    def read_tlim(self):
        
        self.tlim, safety_counter = None, 0
        
        while (self.tlim is None) and (safety_counter<10):
            for key in self.nwbfile.acquisition:
                try:
                    self.tlim = [self.nwbfile.acquisition[key].starting_time,
                                 self.nwbfile.acquisition[key].starting_time+\
                                 (self.nwbfile.acquisition[key].data.shape[0]-1)/self.nwbfile.acquisition[key].rate]
                except BaseException as be:
                    pass
        if self.tlim is None:
            self.tlim = [0, 50] # bad for movies


    def read_data(self):

        # ophys data
        if 'ophys' in self.nwbfile.processing:
            self.read_and_format_ophys_data()
        else:
            for key in ['Segmentation', 'Fluorescence', 'iscell', 'redcell', 'plane',
                        'valid_roiIndices', 'neuropil']:
                setattr(self, key, None)
                
        if 'Pupil' in self.nwbfile.processing:
            self.read_pupil()
            
        if 'FaceMotion' in self.nwbfile.processing:
            self.read_facemotion()
            

    #########################################################
    #       CALCIUM IMAGING DATA (from suite2p output)      #
    #########################################################
    
    def initialize_ROIs(self, 
                        valid_roiIndices=None):

        if valid_roiIndices is None:
            self.nROIs = self.Segmentation.columns[0].data.shape[0]
            self.valid_roiIndices = np.arange(self.nROIs)
            # ---------------------------------------------------------
            # only when they were no previous roi validation
            #         -> we read the table properties
            # ---------------------------------------------------------
            # looping over the table properties (0,1 -> rois locs)
            #      for the ROIS to overwrite the defaults:
            for i in range(2, len(self.Segmentation.columns)):
                if self.Segmentation.columns[i].name=='iscell': # DEPRECATED
                    self.valid_roiIndices = self.valid_roiIndices[\
                            self.Segmentation.columns[i].data[:,0].astype(bool)]
                if self.Segmentation.columns[i].name=='plane':
                    self.planeID = self.valid_roiIndices[\
                            self.Segmentation.columns[i].data[:].astype(int)]
                if self.Segmentation.columns[i].name=='redcell':
                    self.redcell = self.valid_roiIndices[\
                            self.Segmentation.columns[i].data[:,0].astype(bool)]
        else:
            self.nROIs = len(valid_roiIndices)
            self.valid_roiIndices = valid_roiIndices
            if hasattr(self, 'planeID'):
                self.planeID = self.planeID[self.valid_roiIndices]
            if hasattr(self, 'redcell'):
                self.redcell= self.redcell[self.valid_roiIndices]
            
        # initialize rois properties to default values
        self.iscell = np.ones(self.nROIs, dtype=bool) # deprecated
        self.planeID = np.zeros(self.nROIs, dtype=int)
        self.redcell = np.zeros(self.nROIs, dtype=bool) 


    def read_and_format_ophys_data(self):
       
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

        self.initialize_ROIs(valid_roiIndices=None)
                
        
    ######################
    #    LOCOMOTION
    ######################
    def build_running_speed(self,
                            specific_time_sampling=None,
                            interpolation='linear',
                            verbose=False):
        """
        build distance from mean (x,y) position of pupil
        """
        if 'Running-Speed' in self.nwbfile.acquisition:

            self.running_speed = self.nwbfile.acquisition['Running-Speed'].data[:]
            self.t_running_speed = self.nwbfile.acquisition['Running-Speed'].starting_time+\
                np.arange(self.nwbfile.acquisition['Running-Speed'].num_samples)\
                                        /self.nwbfile.acquisition['Running-Speed'].rate

            if specific_time_sampling is not None:
                return tools.resample(self.t_running_speed, 
                                      self.running_speed, 
                                      specific_time_sampling, 
                                      interpolation=interpolation,
                                      verbose=verbose)
        else:
            return None

    
    ######################
    #       PUPIL 
    ######################        

    def read_pupil(self):

        pd = str(self.nwbfile.processing['Pupil'].description)
        if len(pd.split('pix_to_mm='))>1:
            self.FaceCamera_mm_to_pix = int(1./float(pd.split('pix_to_mm=')[-1]))
        else:
            self.FaceCamera_mm_to_pix = 1

    def build_pupil_diameter(self,
                             specific_time_sampling=None,
                             interpolation='linear',
                             verbose=False):
        """
        build pupil diameter trace, i.e. twice the maximum of the ellipse radius at each time point
        """
        if 'Pupil' in self.nwbfile.processing:

            self.t_pupil = self.nwbfile.processing['Pupil'].data_interfaces['cx'].timestamps
            self.pupil_diameter =  2*np.max([self.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:],
                                             self.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:]], axis=0)

            if specific_time_sampling is not None:
                return tools.resample(self.t_pupil, self.pupil_diameter,
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

        else:
            return None


    def build_gaze_movement(self,
                            specific_time_sampling=None,
                            interpolation='linear',
                            verbose=False):
        """
        build distance from mean (x,y) position of pupil
        """

        if 'Pupil' in self.nwbfile.processing:

            self.t_pupil = self.nwbfile.processing['Pupil'].data_interfaces['cx'].timestamps
            cx = self.nwbfile.processing['Pupil'].data_interfaces['cx'].data[:]
            cy = self.nwbfile.processing['Pupil'].data_interfaces['cy'].data[:]
            self.gaze_movement = np.sqrt((cx-np.mean(cx))**2+(cy-np.mean(cy))**2)

            if specific_time_sampling is not None:
                return tools.resample(self.t_pupil, self.gaze_movement, 
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

        else:
            return None
        

    #########################
    #       FACEMOTION  
    #########################      
    
    def read_facemotion(self):
        
        fd = str(self.nwbfile.processing['FaceMotion'].description)
        self.FaceMotion_ROI = [int(i) for i in fd.split('y0,dy)=(')[1].split(')')[0].split(',')]

    def build_facemotion(self,
                         specific_time_sampling=None,
                         interpolation='linear',
                         verbose=False):
        """
        build facemotion
        """

        if 'FaceMotion' in self.nwbfile.processing:

            self.t_facemotion = self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].timestamps
            self.facemotion =  self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].data[:]

            if specific_time_sampling is not None:
                return tools.resample(self.t_facemotion, self.facemotion, 
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

        else:
            return None

    #############################
    #       Calcium Imaging     #
    #############################

    def compute_ROI_indices(self,
                            roiIndex=None,
                            roiIndices='all',
                            verbose=True):

        if roiIndex is not None:
            return roiIndex
        elif roiIndices=='all':
            return np.array(self.valid_roiIndices, dtype=int)
        else:
            return np.array(self.valid_roiIndices[np.array(roiIndices)],\
                    dtype=int)
        
        
    def build_dFoF(self,
                   roiIndex=None, roiIndices='all',
                   roi_to_neuropil_fluo_inclusion_factor=\
                           ROI_TO_NEUROPIL_INCLUSION_FACTOR,
                   neuropil_correction_factor=\
                           NEUROPIL_CORRECTION_FACTOR,
                   method_for_F0=METHOD,
                   percentile=PERCENTILE,
                   sliding_window=T_SLIDING,
                   with_correctedFluo_and_F0=False,
                   specific_time_sampling=None,
                   smoothing=None,
                   interpolation='linear',
                   verbose=True):
        """
        creates self.dFoF, self.t_dFoF
        """

        if not hasattr(self, 'rawFluo'):
            self.build_rawFluo(roiIndex=roiIndex, 
                               roiIndices='all',
                               specific_time_sampling=specific_time_sampling,
                               interpolation=interpolation,
                               verbose=verbose)
        self.t_dFoF = self.t_rawFluo

        if not hasattr(self, 'neuropil'):
            self.build_neuropil(roiIndex=roiIndex, 
                                roiIndices='all',
                                specific_time_sampling=specific_time_sampling,
                                interpolation=interpolation,
                                verbose=verbose)

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
                            verbose=verbose)

    def build_Zscore_dFoF(self, verbose=True):
        """
        /!\ do not deal with specific time sampling /!\ 
        """
        if not hasattr(self, 'dFoF'):
            self.build_dFoF(verbose=verbose)
        setattr(self, 'Zscore_dFoF', (self.dFoF-self.dFoF.mean(axis=0).reshape(1, self.dFoF.shape[1]))/self.dFoF.std(axis=0).reshape(1, self.dFoF.shape[1]))

    def build_Deconvolved(self, Tau=1.3):
        if not hasattr(self, 'dFoF'):
            print('\n deconvolution not possible \n --> need to build_dFoF(**options) first !! ')
        else:
            setattr(self, 'Deconvolved',
                    oasis(self.dFoF, 
                          self.dFoF.shape[0], # batch size
                              Tau, 1./self.CaImaging_dt))


    def build_neuropil(self,
                       roiIndex=None, roiIndices='all',
                       specific_time_sampling=None,
                       interpolation='linear',
                       verbose=True):
        """
        we build the neuropil matrix in the form (nROIs, time_samples)
            we need to deal with the fact that matrix orientation 
            was changed because of pynwb complains
        """
        if self.nROIs==self.Neuropil.data.shape[0]:
            self.neuropil = self.Neuropil.data[\
                    self.compute_ROI_indices(roiIndex=roiIndex,\
                                             roiIndices=roiIndices,\
                                             verbose=verbose),:]
        else:
            # data badly oriented --> transpose in that case
            self.neuropil = np.array(self.Neuropil.data).T[\
                                            self.compute_ROI_indices(\
                                                    roiIndex=roiIndex,
                                                    roiIndices=roiIndices,
                                                    verbose=verbose),:]

        if not hasattr(self, 't_neuropil'):
            self.t_neuropil = self.Neuropil.timestamps[:]

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
        """
        if self.nROIs==self.Fluorescence.data.shape[0]:
            self.rawFluo = self.Fluorescence.data[self.compute_ROI_indices(roiIndex=roiIndex,
                                                                           roiIndices=roiIndices,
                                                                           verbose=verbose), :]
        else:
            # data badly oriented --> transpose in that case
            self.rawFluo = np.array(self.Fluorescence.data).T[self.compute_ROI_indices(roiIndex=roiIndex,
                                                                             roiIndices=roiIndices,
                                                                             verbose=verbose),:]
        if not hasattr(self, 't_rawFluo'):
            self.t_rawFluo = self.Fluorescence.timestamps[:]

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
    
    def init_visual_stim(self, verbose=True):
        self.metadata['load_from_protocol_data'], self.metadata['no-window'] = False, True
        self.metadata['verbose'] = verbose
        self.visual_stim = build_stim(self.metadata)

        
    def get_protocol_id(self, protocol_name):
        cond = np.argwhere(self.protocols==protocol_name).flatten()
        if len(cond)==1:
            return cond[0]
        else:
            print(' /!\\ protocol "%s" not found in data with protocols:' % protocol_name)
            print(self.protocols)
            return None

    
    def get_protocol_cond(self, protocol_id, protocol_name=None):
        """
        ## a recording can have multiple protocols inside
        -> find the condition of a given protocol ID

        'None' to have them all 
        """

        if (protocol_name is not None) and (('protocol_id' in self.nwbfile.stimulus) and\
                (len(np.unique(self.nwbfile.stimulus['protocol_id'].data[:]))>1)):
            protocol_id = self.get_protocol_id(protocol_name)
            Pcond = (self.nwbfile.stimulus['protocol_id'].data[:]==protocol_id)

        elif (protocol_id is not None) and (('protocol_id' in self.nwbfile.stimulus) and\
                (len(np.unique(self.nwbfile.stimulus['protocol_id'].data[:]))>1)):
            Pcond = (self.nwbfile.stimulus['protocol_id'].data[:]==protocol_id)

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
                    cond = cond & (self.nwbfile.stimulus[k].data[Pcond]==xk.flatten()[i])
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

        cond = (time>=self.nwbfile.stimulus[start_key].data[:]) & (time<=self.nwbfile.stimulus[stop_key].data[:])

        if np.sum(cond)>0:
            return np.arange(self.nwbfile.stimulus[start_key].num_samples)[cond][0]
        else:
            return -1

        
    ###########################
    #       other methods     #
    ###########################
    
    def close(self):
        self.io.close()
        
    def list_subquantities(self, quantity):
        if quantity=='CaImaging':
            return ['rawFluo', 'dFoF', 'neuropil', 'Deconvolved',
                    'F-0.7*Fneu', 'F-Fneu', 'd(F-Fneu)', 'd(F-0.7*Fneu)']
        else:
            return ['']
        
            
        
def scan_folder_for_NWBfiles(folder, 
                             sorted_by='filename',
                             Nmax=1000000,
                             exclude_intrinsic_imaging_files=True,
                             verbose=True):
    """
    scan folders for protocols and returns a A

    by default: exccludes the intrinsic imaging files
    """
    if verbose:
        print('inspecting the folder "%s" [...]' % folder)
        t0 = time.time()

    FILES = get_files_with_extension(folder,
                    extension='.nwb', recursive=True)
    
    if exclude_intrinsic_imaging_files:
        FILES = [f for f in FILES if (('left-' not in f) and\
                                      ('down-' not in f) and\
                                      ('right-' not in f) and\
                                      ('up-' not in f))]

    DATES = np.array([f.split(os.path.sep)[-1].split('-')[0] for f in FILES])
    SUBJECTS, PROTOCOLS = [], []

    for f in FILES[:Nmax]:
        try:
            data = Data(f, metadata_only=True, verbose=False)
            PROTOCOLS.append(data.protocols)
            SUBJECTS.append(data.metadata['subject_ID'])
        except BaseException as be:
            SUBJECTS.append('N/A')
            if verbose:
                print(be)
                print('\n /!\\ Pb with "%s" \n' % f)
        
    if verbose:
        print(' -> found n=%i datafiles (in %.1fs) ' % (len(FILES), (time.time()-t0)))

    # sorted by filename

    if sorted_by=='filename':
        isorted = np.argsort(FILES)
    elif sorted_by=='subject':
        isorted = np.argsort(SUBJECTS)
    elif sorted_by=='date':
        isorted = np.argsort(DATES)
    else:
        print(' "%s" no recognized , --> sorted by filename by default ! ' % sorted_by)
        isorted = np.argsort(FILES)

    return {'files':np.array(FILES)[isorted], 
            'dates':np.array(DATES)[isorted],
            'subjects':np.array(SUBJECTS)[isorted],
            'protocols':[PROTOCOLS[i] for i in isorted]}
