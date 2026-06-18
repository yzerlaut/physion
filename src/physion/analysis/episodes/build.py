import sys, time, os, pathlib, string, itertools

import numpy as np
from scipy.interpolate import interp1d

from physion.analysis import stat_tools
from physion.visual_stim import stimuli
from physion.visual_stim.build import get_default_params
import physion.analysis.episodes as episodes
from .trial_statistics import pre_post_statistics,\
        stat_test_for_evoked_responses, reliability

class EpisodeData:
    """
    Object to Analyze Responses to several Episodes
        - episodes should have the same durations
        - episodes can correspond to diverse stimulation parameters

    - Using the photodiode-signal-derived timestamps
            to build "episode response" by interpolating
            the raw signal on a fixed time interval (surrounding the stim)

    - Using metadata to store stimulus informations per episode

    quantities should be one of:
        'photodiode',
        'pupil',
        'facemotion',
        'running',
        'opto',
        'rawFluo',
        'neuropil',
        'dFoF',
        'spikes',
        'LFP',
        'MUA'
    """

    def __init__(self, full_data,
                 protocol_id=None, protocol_name=None,
                 quantities=['photodiode'],
                 quantities_args=None,
                 prestim_duration=None, # to force the prestim window otherwise, half the value in between episodes
                 dt_sampling=1, # ms
                 interpolation='linear',
                 tfull=None,
                 verbose=False):
        '''
        Initializes EpisodeData

        '''

        self.dt_sampling = dt_sampling
        self.verbose = verbose
        self.data = full_data

        self.select_protocol_from(full_data,
                                  protocol_id=protocol_id,
                                  protocol_name=protocol_name)

        ################################################
        #          Episode Calculations
        ################################################

        self.set_quantities(full_data, quantities,
                            quantities_args=quantities_args,
                            prestim_duration=prestim_duration,
                            dt_sampling=dt_sampling,
                            interpolation=interpolation,
                            tfull=tfull,
                            verbose=verbose)

        ################################################i
        #           some clean up
        ################################################i
        # because "protocol_id" and "protocol_name" are over-written by self.set_quantities

        if (protocol_id is None):
            if verbose:
                print("Protocol ID is None")
            if (protocol_name is not None):
                if verbose:
                    print("Protocol name is not None -> get protocol ID")
                protocol_id = full_data.get_protocol_id(protocol_name)
            else:
                print("  [!!] protocol_name & protocol_id not specified [!!] ")
                print("          --> taking the first protocol of protocol_id=0")
                protocol_id = 0

        # we overwrite those to single values
        self.protocol_id = protocol_id
        self.protocol_name = full_data.protocols[self.protocol_id]

        if self.verbose:
            print('  -> [ok] episodes ready !')
        #move this to set_quantities?#

    ##############################################
    ###########   --- METHODS ---  ###############
    ##############################################

    def select_protocol_from(self, full_data,
                             protocol_id=None,
                             protocol_name=None):
        """
        Creates self.protocol_cond_in_full_data to filter the full data to this one protocol
        self.protocol_cond_in_full_data is an ndarray of bool
        Does not return anything

        """
        # choose protocol
        if (protocol_id is None) and (protocol_name is not None):
            protocol_id = full_data.get_protocol_id(protocol_name)
        
        elif (protocol_id is None) and (protocol_name is None):
            protocol_id = 0
            print('protocols:', full_data.protocols)
            print(' [!!] need to explicit the "protocol_id" or "protocol_name" [!!] ')
            print('         ---->   set to protocol_id=0 by default \n ')
        
        self.protocol_cond_in_full_data = \
            full_data.get_protocol_cond(protocol_id)

    def set_quantities(self, full_data, quantities,
                       quantities_args=None,
                       prestim_duration=None,
                       dt_sampling=1, # ms
                       interpolation='linear',
                       tfull=None,
                       verbose=True):
       
        """
        Sets self.varied_parameters and self.fixed_parameters
        Sets time self.t
        Sets each self.parameter given full_data.nwbfile.stimulus.keys() as np.arrays
        Sets each self.quantity given by argument as np.arrays
        Sets self.index_from_start (the first indexes where the protocol condition is satified = when protocol starts)
        Sets self.quantities  as a list of str 

        """
        if quantities_args is None:
            quantities_args = [{} for q in quantities]
        for q in quantities_args:
            q['verbose'] = self.verbose

        if self.verbose:
            print('  Number of episodes over the whole recording: %i/%i (with protocol condition)' % (np.sum(self.protocol_cond_in_full_data), len(self.protocol_cond_in_full_data)))
            print('  building episodes with %i modalities [...]' % len(quantities))

        # find the parameter(s) varied within that specific protocol
        self.varied_parameters, self.fixed_parameters =  {}, {}

        for key in full_data.nwbfile.stimulus.keys():
            if key not in ['frame_run_type', 'index', 'protocol_id',
                           'time_duration', 'time_start',
                           'time_start_realigned', 'time_stop',
                           'time_stop_realigned', 'interstim',
                           'protocol-name', 'OptogeneticSeries']:
                unique = np.sort(np.unique(full_data.nwbfile.stimulus[key].data[self.protocol_cond_in_full_data,0]))
                if len(unique)>1:
                    self.varied_parameters[key] = unique
                elif len(unique)==1:
                    self.fixed_parameters[key] = unique

        # new sampling, a window arround stimulus presentation
        if (prestim_duration is None) and ('interstim' in full_data.nwbfile.stimulus):
            prestim_duration = np.min(full_data.nwbfile.stimulus['interstim'].data[:,0])/2. # half the stim duration
        if (prestim_duration is None) or (prestim_duration<1):
            prestim_duration = 1 # still 1s is a minimum
        ipre = int(prestim_duration/dt_sampling*1e3)

        duration = full_data.nwbfile.stimulus['time_stop'].data[self.protocol_cond_in_full_data,0][0]-\
                full_data.nwbfile.stimulus['time_start'].data[self.protocol_cond_in_full_data,0][0]
        idur = int(duration/dt_sampling/1e-3)
        # -> time array:
        self.t = np.arange(-ipre+2, idur+ipre)*dt_sampling*1e-3


        #############################################################################
        ############ we do it modality by modality (quantity)  ######################
        #############################################################################

        QUANTITIES, QUANTITY_VALUES, QUANTITY_TIMES = [], [], []

        for iq, quantity, quantity_args in zip(range(len(quantities)), quantities, quantities_args):

            if type(quantity)!=str and (tfull is not None):
                # if quantity is an array and not a keyword
                QUANTITY_VALUES.append(quantity)
                QUANTITY_TIMES.append(tfull)
                QUANTITIES.append('quant_%i' % iq)

            elif hasattr(full_data, quantity):
                # quantity is a string and it's already built
                QUANTITY_VALUES.append(getattr(full_data, quantity))
                QUANTITY_TIMES.append(getattr(full_data, 't_%s' % quantity))
                QUANTITIES.append(quantity)

            else:
                # quantity is a string but not already built
                try:
                    if verbose:
                        print('           ->  building %s [...]' % quantity)
                    getattr(full_data, 'build_%s' % quantity)(**quantity_args)
                    QUANTITY_VALUES.append(getattr(full_data, quantity))
                    QUANTITY_TIMES.append(getattr(full_data, 't_%s' % quantity))
                    QUANTITIES.append(quantity)
                except BaseException as be:
                    print()
                    print(be)
                    print()
                    print(30*'-')
                    print('[!!] Pb, ', quantity, 'not recognized')
                    print(30*'-')


        # adding the parameters
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, [])

        for q in QUANTITIES:
            setattr(self, q, [])

        for iEp in np.arange(full_data.nwbfile.stimulus['time_start'].num_samples)[self.protocol_cond_in_full_data]:

            tstart = full_data.nwbfile.stimulus['time_start_realigned'].data[iEp,0]
            tstop = full_data.nwbfile.stimulus['time_stop_realigned'].data[iEp,0]

            # print(iEp, tstart, tstop)
            # print(full_data.nwbfile.stimulus['patch-delay'].data[iEp])

            RESPS, success = [], True
            for quantity, tfull, valfull in zip(QUANTITIES, QUANTITY_TIMES, QUANTITY_VALUES):

                # compute time and interpolate
                ep_cond = (tfull>=(tstart-2.*prestim_duration)) & (tfull<(tstop+1.5*prestim_duration)) # higher range of interpolation to avoid boundary problems
                try:
                    if (len(valfull.shape)>1):
                        # multi-dimensional response, e.g. dFoF = (rois, time)
                        resp = np.zeros((valfull.shape[0], len(self.t)))
                        for j in range(valfull.shape[0]):
                            func = interp1d(tfull[ep_cond]-tstart, valfull[j,ep_cond],
                                            kind=interpolation, bounds_error=False,
                                            fill_value=(valfull[j,ep_cond][0], valfull[j,ep_cond][-1]))
                            resp[j, :] = func(self.t)
                        RESPS.append(resp)

                    else:
                        func = interp1d(tfull[ep_cond]-tstart, valfull[ep_cond],
                                        kind=interpolation)
                        RESPS.append(func(self.t))

                except BaseException as be:

                    success=False # we switch this off to remove the episode in all modalities
                    if self.verbose:
                        print('----')
                        print(be)
                        # print(tfull[ep_cond][0]-tstart, tfull[ep_cond][-1]-tstart, tstop-tstart)
                        print(quantity)
                        print('Problem with episode %i between (%.2f, %.2f)s' % (iEp, tstart, tstop))


            if success:

                # only succesful episodes in all modalities
                for quantity, response in zip(QUANTITIES, RESPS):
                    getattr(self, quantity).append(response)
                for key in full_data.nwbfile.stimulus.keys():
                    try:
                        getattr(self, key).append(full_data.nwbfile.stimulus[key].data[iEp,0])
                    except BaseException as be:
                        pass # we skip thise variable

        # transform stim params to np.array
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, np.array(getattr(self, key)))

        for q in QUANTITIES:
            setattr(self, q, np.array(getattr(self, q)))

        self.quantities = QUANTITIES

    def get_response2D(self, 
                     quantity=None, 
                     episode_cond=None,
                     index=None,
                     averaging_dimension='ROIs'):
        """
        takes the quantity you want the response from (default will be the first one). Check with ep.quantities
        can take conditions on episodes, roi index (value, array of values or None), and average dimension (ROIs or episodes, default episodes)
        
        Makes an average of rois or episodes (with condition filter) to obtain a 2D matrix 
        
        returns a tuple, two dimensional matrix, 1 dim = each episode OR each roi, 2 dim = quantity values across time
        
        single-episode responses can have different shapes
        e.g. 
            self.pupil_diameter.shape() = (Nepisodes, Ntimestamps)
            self.dFoF.shape() = (Nepisodes, Nrois, Ntimestamps)

        index can be either a index, an array of indices or None (default: then all indices)
        """

        if index is None:
            index = np.arange(getattr(self, quantity).shape[1])
            
        if quantity is None:
            if len(self.quantities)>1:
                print('\n there are several modalities in that episode')
                print('     -> need to define the desired quantity, here taking: "%s"' % self.quantities[0])
            quantity = self.quantities[0]

        if episode_cond is None:
            # by default all True
            episode_cond = self.find_episode_cond()

        if len(getattr(self, quantity).shape)==2:  #2 dimensions 
            # i.e. self.quantity.shape = (Nepisodes, Ntimestamps)
            #Filter by episode
            return getattr(self, quantity)[episode_cond, :]

        elif len(getattr(self, quantity).shape)==3: #3 dimensions
            # i.e. self.quantity.shape = (Nepisodes, Nrois, Ntimestamps) 
            # then two cases:
            if type(index) in [int, np.int16, np.int64]:
                return getattr(self, quantity)[episode_cond,index,:]

            else:  #index is an array -> multiple ROIs and multiple episodes
                #could be written in a shorter way
                if averaging_dimension=='episodes':
                    dim = 0
                    return getattr(self, quantity)[episode_cond,:,:].mean(axis=dim)[index,:]
                elif averaging_dimension=='ROIs':
                    dim = 1
                    return getattr(self, quantity)[:,index,:].mean(axis=dim)[episode_cond,:]
                else:
                    print('dimension not recognized, using episodes by default')
                    dim = 0
                    return getattr(self, quantity)[episode_cond,:,:].mean(axis=dim)[index,:]

    def compute_interval_cond(self, interval):
        """
        returns a list of bool, False when t is not in interval, True when it is in interval
        (very useful to define a condition (pre_cond  = self.compute_interval_cond(interval_pre)) to then filter the response (response[:,pre_cond])
        """
        return (self.t>=interval[0]) & (self.t<=interval[1])

    def find_episode_cond(self, key=None, index=None, value=None):
        """
        Conditions can be key (check ep.varied_parameters), 
                          index (which option of the varied parameters) or 
                          value (actual value parameter)
        'key' and 'index' can be either lists of values

        Returns a list of bool, False when episode does not meet conditions, True if passes conditions. Size # episodes
        By default no condition, all True

        example for a 

        data.find_episode_cond(key='angle', index=0)

        data.find_episode_cond(key='angle', value=90.0)
        
        """

        cond = np.ones(len(self.time_start), dtype=bool)

        if (type(key) in [list, np.ndarray, tuple]) and\
                (type(index) in [list, np.ndarray, tuple]):
            for n in range(len(key)):
                if key[n] != '':
                    cond = cond & (getattr(self, key[n])==self.varied_parameters[key[n]][index[n]])

        elif (type(key) in [list, np.ndarray, tuple]) and\
                 (type(value) in [list, np.ndarray, tuple]):
            for n in range(len(key)):
                if key[n] != '':
                    cond = cond & (getattr(self, key[n])==value[n])

        elif (key is not None) and (key!='') and (index is not None):
            cond = cond & (getattr(self, key)==self.varied_parameters[key][index])

        elif (key is not None) and (key!='') and (value is not None):
            cond = cond & (getattr(self, key)==value)

        if np.sum(cond)==0:
            print('all false, not match found')

        return cond

    def plot_stim_picture(self, 
                         iEp=0,
                         ax=None,
                         force_degree=False,
                         **args):
        """
        ###########################################################
        we rebuild an artificial **single stimulus** protocol 
            for the stimulus parameters of that episode

        """
        stim_data = {'no-window':True,
                     'Screen' : self.data.metadata['Screen'],
                     'Stimulus': self.data.metadata['Stimulus']}

        default_params = get_default_params(stim_data['Stimulus'])
        # replacing the default params with those of this episode
        for key in default_params:
            if key in self.varied_parameters:
                stim_data[key] = getattr(self, key)[iEp]
            elif 'Protocol-%i-%s' % (self.protocol_id+1, key) in self.data.metadata:
                stim_data[key] = self.data.metadata['Protocol-%i-%s' % (self.protocol_id+1, key)]
            elif key in self.data.metadata:
                stim_data[key] = self.data.metadata[key]

        if force_degree:
            stim_data['units'] = 'deg'
        stim_data['Presentation'] = 'Single-Stimulus'

        # we generate a new visual_stim object from this single episode info
        visual_stim = getattr(\
                         getattr(\
                                stimuli,\
                                    stim_data['Stimulus']),
                                              'stim')(stim_data)

        # we use the plot_stim_picture method from visual stim
        visual_stim.plot_stim_picture(0, ax=ax, **args)
        
        return 0
    
    def pre_post_statistics(self, **args):
        return pre_post_statistics(self, **args)
    def stat_test_for_evoked_responses(self, **args):
        return stat_test_for_evoked_responses(self, **args)
    def reliability(self, **args):
        return reliability(self, **args)


if __name__=='__main__':

    from physion.analysis.read_NWB import Data
    from physion.utils import plot_tools as pt

    filename = sys.argv[-1]
    data= Data(filename)
    data.build_running()

    print("Protocols : ", data.protocols)

    print("Protocol name : ", data.protocols[0])
    ep = EpisodeData(data, quantities=['running'], protocol_id=0)
    print("Varied parameters : ", ep.varied_parameters)

    # summary = ep.pre_post_statistics(\
    #     response_args=dict(quantity='running'))

    for key in ep.varied_parameters:
        print(getattr(ep, key)[:3])

    for i in range(3):
        fig, ax = pt.figure(ax_scale=(2,2))
        ep.plot_stim_picture(i, ax=ax)
        pt.plt.show()


        

