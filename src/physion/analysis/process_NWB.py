import sys, time, os, pathlib, string, itertools

import numpy as np
from scipy.stats import sem
from scipy.interpolate import interp1d

from physion.analysis import stat_tools
from physion.visual_stim.build import build_stim

class EpisodeData:
    """
    Object to Analyze Responses to several Episodes
        - episodes should have the same durations
        - episodes can correspond to diverse stimulation parameters

    - Using the photodiode-signal-derived timestamps
            to build "episode response" by interpolating
            the raw signal on a fixed time interval (surrounding the stim)

    - Using metadata to store stimulus informations per episode

    quantities should be given as:
            - Photodiode-Signal
            - running_speed
            - Deconvolved
            - dFoF
            - Zscore_dFoF
            - neuropil
            - rawFluo
            - pupil_diameter
            - gaze_movement
            - facemotion
    """

    def __init__(self, full_data,
                 protocol_id=None, protocol_name=None,
                 quantities=['Photodiode-Signal'],
                 quantities_args=None,
                 prestim_duration=None, # to force the prestim window otherwise, half the value in between episodes
                 dt_sampling=1, # ms
                 interpolation='linear',
                 with_visual_stim=False,
                 tfull=None,
                 verbose=True):
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
                            tfull=tfull)

        ################################################i
        #           some clean up
        ################################################i
        # because "protocol_id" and "protocol_name" are over-written by self.set_quantities

        if (protocol_id is None):
            # print("Protocol ID is None")
            if (protocol_name is not None):
                # print("Protocol name is not None -> get protocol ID")
                protocol_id = full_data.get_protocol_id(protocol_name)
            else:
                print("""
                      protocol_id and protocol_name are both None,
                            --> setting to protocol_id=0 (CHECK THAT IT IS WHAT YOU WANT)
                      """)
                protocol_id = 0

        # we overwrite those to single values
        self.protocol_id = protocol_id
        self.protocol_name = full_data.protocols[self.protocol_id]

        # VISUAL STIM
        if with_visual_stim:
            self.init_visual_stim(full_data)
        else:
            self.visual_stim = None

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
        

        self.protocol_cond_in_full_data = full_data.get_protocol_cond(protocol_id)

    def set_quantities(self, full_data, quantities,
                       quantities_args=None,
                       prestim_duration=None,
                       dt_sampling=1, # ms
                       interpolation='linear',
                       tfull=None):
       
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
                           'protocol-name']:
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
                QUANTITY_VALUES.append(quantity)
                QUANTITY_TIMES.append(tfull)
                QUANTITIES.append('quant_%i' % iq)

            elif quantity in ['running_speed', 'Running-Speed', 'RunningSpeed', 'Locomotion']:
                if not hasattr(full_data, 'running_speed'):
                    full_data.build_running_speed(**quantity_args)
                QUANTITY_VALUES.append(full_data.running_speed)
                QUANTITY_TIMES.append(full_data.t_running_speed)
                QUANTITIES.append('running_speed')

            elif quantity in ['Deconvolved']:
                if not hasattr(full_data, 'Deconvolved'):
                    full_data.build_Deconvolved(**quantity_args)
                QUANTITY_VALUES.append(full_data.Deconvolved)
                QUANTITY_TIMES.append(full_data.t_dFoF)
                QUANTITIES.append('Deconvolved')

            elif quantity in ['dFoF', 'dF/F']:
                if not hasattr(full_data, 'dFoF'):
                    full_data.build_dFoF(**quantity_args)
                QUANTITY_VALUES.append(full_data.dFoF)
                QUANTITY_TIMES.append(full_data.t_dFoF)
                QUANTITIES.append('dFoF')

            elif quantity in ['Zscore_dFoF', 'Zscore_dF/F']:
                if not hasattr(full_data, 'Zscore_dFoF'):
                    full_data.build_Zscore_dFoF(**quantity_args)
                QUANTITY_VALUES.append(full_data.Zscore_dFoF)
                QUANTITY_TIMES.append(full_data.t_dFoF)
                QUANTITIES.append('Zscore_dFoF')

            elif quantity in ['Neuropil', 'neuropil']:
                if not hasattr(full_data, 'neuropil'):
                    full_data.build_neuropil(**quantity_args)
                QUANTITY_VALUES.append(full_data.neuropil)
                QUANTITY_TIMES.append(full_data.t_neuropil)
                QUANTITIES.append('neuropil')

            elif quantity in ['Fluorescence', 'rawFluo']:
                if not hasattr(full_data, 'rawFluo'):
                    full_data.build_rawFluo(**quantity_args)
                QUANTITY_VALUES.append(full_data.rawFluo)
                QUANTITY_TIMES.append(full_data.t_rawFluo)
                QUANTITIES.append('rawFluo')

            elif quantity in ['pupil_diameter', 'Pupil', 'pupil-size', 'Pupil-diameter', 'pupil-diameter', 'pupil']:
                if not hasattr(full_data, 'pupil_diameter'):
                    full_data.build_pupil_diameter(**quantity_args)
                QUANTITY_VALUES.append(full_data.pupil_diameter)
                QUANTITY_TIMES.append(full_data.t_pupil)
                QUANTITIES.append('pupil_diameter')

            elif quantity in ['gaze', 'Gaze', 'gaze_movement', 'gazeMovement', 'gazeDirection']:
                if not hasattr(full_data, 'gaze_movement'):
                    full_data.build_gaze_movement(**quantity_args)
                QUANTITY_VALUES.append(full_data.gaze_movement)
                QUANTITY_TIMES.append(full_data.t_pupil)
                QUANTITIES.append('gaze_movement')

            elif quantity in ['facemotion', 'FaceMotion', 'faceMotion']:
                if not hasattr(full_data, 'facemotion'):
                    full_data.build_facemotion(**quantity_args)
                QUANTITY_VALUES.append(full_data.facemotion)
                QUANTITY_TIMES.append(full_data.t_facemotion)
                QUANTITIES.append('faceMotion')

            else:
                if quantity in full_data.nwbfile.acquisition:
                    QUANTITY_TIMES.append(np.arange(full_data.nwbfile.acquisition[quantity].data.shape[0])/full_data.nwbfile.acquisition[quantity].rate)
                    QUANTITY_VALUES.append(full_data.nwbfile.acquisition[quantity].data[:,0])
                    QUANTITIES.append(full_data.nwbfile.acquisition[quantity].name.replace('-', '').replace('_', ''))
                elif quantity in full_data.nwbfile.processing:
                    QUANTITY_TIMES.append(np.arange(full_data.nwbfile.processing[quantity].data.shape[0])/full_data.nwbfile.processing[quantity].rate)
                    QUANTITY_VALUES.append(full_data.nwbfile.processing[quantity].data[:,0])
                    QUANTITIES.append(full_data.nwbfile.processing[quantity].name.replace('-', '').replace('_', ''))
                else:
                    print(30*'-')
                    print(quantity, 'not recognized')
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
                     roiIndex=None, 
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

        roiIndex can be either a index, an array of indices or None (default: then all indices)
        """


        if roiIndex is None:
            roiIndex = np.arange(self.data.nROIs)  
            
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
            if type(roiIndex) in [int, np.int16, np.int64]:
                return getattr(self, quantity)[episode_cond,roiIndex,:]

            else:  #roiIndex is an array -> multiple ROIs and multiple episodes
                #could be written in a shorter way
                if averaging_dimension=='episodes':
                    dim = 0
                    return getattr(self, quantity)[episode_cond,:,:].mean(axis=dim)[roiIndex,:]
                elif averaging_dimension=='ROIs':
                    dim = 1
                    return getattr(self, quantity)[:,roiIndex,:].mean(axis=dim)[episode_cond,:]
                else:
                    print('dimension not recognized, using episodes by default')
                    dim = 0
                    return getattr(self, quantity)[episode_cond,:,:].mean(axis=dim)[roiIndex,:]

    def compute_interval_cond(self, interval):
        """
        returns a list of bool, False when t is not in interval, True when it is in interval
        (very useful to define a condition (pre_cond  = self.compute_interval_cond(interval_pre)) to then filter the response (response[:,pre_cond])
        """
        return (self.t>=interval[0]) & (self.t<=interval[1])

    def find_episode_cond(self, key=None, index=None, value=None):
        """
        Conditions can be key (check ep.varied_parameters), index (which option of the varied parameters) or value (??)
        'key' and 'index' can be either lists of values

        Returns a list of bool, False when episode does not meet conditions, True if passes conditions. Size # episodes
        By default no condition, all True

        example for a 

        data.find_episode_cond(key='orientation', index=0)

        data.find_episode_cond(key='orientation', value=90.0)
        
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

    def stat_test_for_evoked_responses(self,
                                       episode_cond=None,
                                       response_args={},
                                       interval_pre=[-2,0], interval_post=[1,3],
                                       test='wilcoxon',
                                       sign='positive',
                                       verbose=True):
        """
        Takes EpisodeData
        Choose quantity from where you want to do a statistical test. Check possibilities with ep.quantities
        Choose the test you want . default wilcoxon . 

        It performs a test between the values from interval_pre and interval_post
        
        returns pvalue and statistic 
        """

        response = self.get_response2D(episode_cond = episode_cond,
                                       **response_args)

        pre_cond  = self.compute_interval_cond(interval_pre)
        post_cond  = self.compute_interval_cond(interval_post)

        # print(response[episode_cond,:][:,pre_cond].mean(axis=1))
        # print(response[episode_cond,:][:,post_cond].mean(axis=1))
        # print(len(response.shape)>1,(np.sum(episode_cond)>1))

        if len(response.shape)>1:
            return stat_tools.StatTest(response[:,pre_cond].mean(axis=1),
                                       response[:,post_cond].mean(axis=1),
                                       test=test, 
                                       sign=sign,
                                       verbose=verbose)
        else:
            return stat_tools.StatTest(None, None,
                                       test=test, sign=sign,
                                       verbose=verbose)

    def compute_summary_data(self, stat_test_props,
                             episode_cond=None,
                             exclude_keys=['repeat'],
                             response_args={},
                             response_significance_threshold=0.01,
                             verbose=True):
        '''
        return all the statistic values organized in a dictionary (str keys and arrays of values). 
        dictionnary keys : 'value', 'std-value',  'sem-value', 'significant', 'relative_value', 'angle', 'angle-index', 'angle-bins'
        '''

        if episode_cond is None:
            episode_cond = self.find_episode_cond() # all true by default

        VARIED_KEYS, VARIED_VALUES, VARIED_INDICES, Nfigs, VARIED_BINS = [], [], [], 1, []
        for key in self.varied_parameters:
            if key not in exclude_keys:
                VARIED_KEYS.append(key)
                VARIED_VALUES.append(self.varied_parameters[key])
                VARIED_INDICES.append(np.arange(len(self.varied_parameters[key])))
                x = np.unique(self.varied_parameters[key])
                VARIED_BINS.append(np.concatenate([[x[0]-.5*(x[1]-x[0])],
                                                   .5*(x[1:]+x[:-1]),
                                                   [x[-1]+.5*(x[-1]-x[-2])]]))

        summary_data = {'value':[], 'std-value':[], 'sem-value':[], 
                        'significant':[], 'relative_value':[]}

        for key, bins in zip(VARIED_KEYS, VARIED_BINS):
            summary_data[key] = []
            summary_data[key+'-index'] = []
            summary_data[key+'-bins'] = bins

        if len(VARIED_KEYS)>0:

            for indices in itertools.product(*VARIED_INDICES):
                stats = self.stat_test_for_evoked_responses(episode_cond=self.find_episode_cond(VARIED_KEYS,
                                                                                                list(indices)) &\
                                                                          episode_cond,
                                                            response_args=response_args,
                                                            verbose=verbose,
                                                            **stat_test_props)
                for key, index in zip(VARIED_KEYS, indices):
                    summary_data[key].append(self.varied_parameters[key][index])
                    summary_data[key+'-index'].append(index)
                # if (stats.x is not None) and (stats.y is not None):
                if stats.r!=0:
                    summary_data['value'].append(np.mean(stats.y-stats.x))
                    summary_data['std-value'].append(np.std(stats.y-stats.x))
                    summary_data['sem-value'].append(sem(stats.y-stats.x))
                    if np.sum(stats.x==0)==0:
                        summary_data['relative_value'].append(np.mean((stats.y-stats.x)/stats.x))
                    else:
                        summary_data['relative_value'].append(np.nan) # if one of them is 0, all to Nan so that it's unusable
                    summary_data['significant'].append(stats.significant(threshold=response_significance_threshold))
                else:
                    for kk in ['value', 'std-value', 'sem-value', 
                               'significant', 'relative_value']:
                        summary_data[kk].append(np.nan)
        else:
            stats = self.stat_test_for_evoked_responses(response_args=response_args,
                                                        **stat_test_props)

            # if (stats.x is not None) and (stats.y is not None):
            if stats.r!=0:
                summary_data['value'].append(np.mean(stats.y-stats.x))
                summary_data['std-value'].append(np.std(stats.y-stats.x))
                summary_data['sem-value'].append(sem(stats.y-stats.x))
                summary_data['significant'].append(stats.significant(threshold=response_significance_threshold))
                summary_data['relative_value'].append(np.mean((stats.y-stats.x)/stats.x))
            else:
                for kk in ['value', 'std-value', 'sem-value',
                           'significant', 'relative_value']:
                    summary_data[kk].append(np.nan)

        for key in summary_data:
            summary_data[key] = np.array(summary_data[key])

        return summary_data

    def init_visual_stim(self, full_data):
        """
        Initialize visual stimulation specific to those episodes (self.visual_stim) by creating a dictionnary from the metadata

        PROBLEM - CHECK WHY THE visual_stim CAN HAVE DIFFERENT VALUES THAN THE DATA
        
        No return output

        start from
        data.init_visual_stim()
        data.visual_stim.get_image(episode_number_in_multiprotocol)

        Ep.init_visual_stim()
        Ep.visual_stim.get_image(episode_number_in_single_protocol)

        test in dataviz.episodes.trial_average.plot

        """

        stim_data = {'no-window':True}

        for key in full_data.metadata:
            stim_data[key]=full_data.metadata[key]
            # if subprotocol, removes the "Protocol-i" from the key
            if ('Protocol-%i-' % (self.protocol_id+1)) in key:
                stim_data[key.replace('Protocol-%i-' % (self.protocol_id+1), '')]=full_data.metadata[key]

        self.visual_stim = build_stim(stim_data)

        # we overwrite the episode values to force it to those of the recording:
        ##
        ## CHECK WHY THE visual_stim CAN HAVE DIFFERENT VALUES THAN THE DATA
        ## is it the seed that is not passed well ?
        ## (not crucial because it is just for stim visualization purpose, but still...)
        ##
        for key in self.visual_stim.experiment:
            if hasattr(self, key):
                self.visual_stim.experiment[key] = getattr(self, key)


if __name__=='__main__':

    import physion

    filename = sys.argv[-1]
    data= physion.analysis.read_NWB.Data(filename)
    data.build_dFoF()
    Episodes = EpisodeData(data, quantities=['dFoF'], protocol_id=2)
    for ia, angle in enumerate(Episodes.varied_parameters['angle']):
        ep_cond = Episodes.find_episode_cond('angle', ia)
        stats = Episodes.stat_test_for_evoked_responses(\
                                episode_cond=ep_cond,
                                response_args={'quantity':'dFoF', 
                                               'roiIndex':3})
        print(ia, angle, stats.significant())

