"""
visual stimulation (photodiode, stimulus parameters, protocols, episodes)

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
import numpy as np
from physion.visual_stim.build import build_stim
from physion.analysis import tools


class VisualStimMixin:

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
