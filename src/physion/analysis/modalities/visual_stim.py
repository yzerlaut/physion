"""
visual stimulation (photodiode, stimulus parameters, protocols, episodes)

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
import numpy as np
import json, re
from physion.visual_stim.build import build_stim, image_key, vse_key, VISUAL_STIM_IMAGES
from physion.visual_stim.stimuli.images import FROM_RECORDING
from physion.analysis import tools


def same_vse(vse1, vse2):
    """ same saccade times, same shifts, same units """
    return (vse1.get('units', 'pixels')==vse2.get('units', 'pixels')) and\
        all(len(vse1[key])==len(vse2[key]) and np.allclose(vse1[key], vse2[key])\
                for key in ['t', 'x', 'y'])


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
            #   (the image files can be missing: images stored in the file, see below)
            flags = [FROM_RECORDING]+['Protocol-%i-%s' % (i, FROM_RECORDING)\
                        for i in range(1, 1+len([k for k in self.metadata\
                                                 if re.fullmatch(r'Protocol-\d+', k)]))]
            self.metadata.update({flag:True for flag in flags})
            try:
                self.visual_stim = build_stim(protocol=self.metadata)
            finally:
                for flag in flags:
                    self.metadata.pop(flag)

            self.set_visual_stim_data_from_file()

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

    def set_visual_stim_data_from_file(self):
        """
        what was really shown (NWB file) instead of what the current code and files give:
            - the virtual saccadic explorations (VSE), not regenerated from the protocol
            - the images, if the image files are missing
        """
        STIM = self.visual_stim.STIM if hasattr(self.visual_stim, 'STIM')\
                    else [self.visual_stim]

        for protocol_id, stim in enumerate(STIM):

            vse = getattr(stim, 'vse', None)
            stored = self.get_stored_vse(protocol_id)
            if isinstance(vse, dict) and (stored is not None):
                if not same_vse(vse, stored):
                    print(' [!!] protocol %i: the VSE regenerated from the protocol differs from' % protocol_id)
                    print('      the one stored in the NWB file --> using the stored one [!!]')
                stim.vse = stored

            if hasattr(stim, 'set_stored_images') and (stim.NIarray is None):
                images = self.get_stored_images(protocol_id)
                stim.set_stored_images(images)
                print(' [!!] protocol %i: image files not available' % protocol_id)
                print('      --> using the %i images stored in the NWB file [!!]' % len(images))

    def get_stored_vse(self, protocol_id):
        """ VSE of protocol "protocol_id" stored in the NWB file (one per protocol), or None """
        prefix = vse_key(protocol_id, 0)[:-1]
        for key in self.nwbfile.stimulus_template:
            if key.startswith(prefix):
                return self.get_visual_stim_vse(protocol_id, seed=int(key[len(prefix):]))
        return None

    def get_stored_images(self, protocol_id):
        """ {Image-ID: image} of protocol "protocol_id" stored in the NWB file """
        if VISUAL_STIM_IMAGES not in self.nwbfile.stimulus_template:
            return {}
        prefix = image_key(protocol_id, 0)[:-1]
        image_ids = [int(key[len(prefix):])\
                        for key in self.nwbfile.stimulus_template[VISUAL_STIM_IMAGES].images\
                            if key.startswith(prefix)]
        return {image_id:self.get_visual_stim_image(protocol_id, image_id)\
                    for image_id in image_ids}

    def get_protocol_id(self, protocol_name):
        cond = np.argwhere(self.protocols==protocol_name).flatten()
        if len(cond)==1:
            return cond[0]
        else:
            print(' [!!] protocol "%s" not found in data with protocols:' % protocol_name)
            print(self.protocols)
            return None

    def get_visual_stim_image(self, protocol_id, image_id):
        """
        image "Image-ID" displayed in protocol "protocol_id" (natural images, ...)
            greyscale (uint8, 0-255), oriented as the movie frames,
            before contrast, background and mask

        None if the image is not stored in the NWB file
        """
        if (VISUAL_STIM_IMAGES in self.nwbfile.stimulus_template) and\
                (image_key(protocol_id, image_id) in\
                    self.nwbfile.stimulus_template[VISUAL_STIM_IMAGES].images):
            return np.array(self.nwbfile.stimulus_template[VISUAL_STIM_IMAGES]\
                                .images[image_key(protocol_id, image_id)].data[:])
        return None

    def get_visual_stim_vse(self, protocol_id, seed=0):
        """
        virtual saccadic exploration (VSE) of protocol "protocol_id" (generated with "seed")
            dictionary as in visual_stim/stimuli/virtual_scene_exploration.py:
            't': saccade times (s, from the episode start),
            'x', 'y': shifts after each saccade, 'max_amplitude',
            'units': 'deg' (visual degrees) or 'pixels' (legacy protocols)

        None if the VSE is not stored in the NWB file
        """
        if vse_key(protocol_id, seed) in self.nwbfile.stimulus_template:
            series = self.nwbfile.stimulus_template[vse_key(protocol_id, seed)]
            data = np.array(series.data[:])
            vse = {'t':np.array(series.timestamps[:]),
                   'x':data[:,0], 'y':data[:,1], 'units':'pixels'}
            vse.update(json.loads(series.comments))
            return vse
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
