import os, sys, itertools, pandas
import numpy as np

from physion.analysis import read_NWB, process_NWB

def Normalize(x):
    if np.std(x)>0:
        return (x-np.mean(x))/np.std(x)
    else:
        return np.zeros(len(x))

def NWB_to_dataframe(nwbfile,
                     visual_stim_label='per-protocol',
                     exclude_from_stim=[],
                     exclude_from_timepoints=['grey-10min'],
                     prestim_duration=0.5,
                     poststim_duration=1.5,
                     time_sampling_reference='dFoF',
                     normalize=[], # array of quantities to normalize
                     subsampling=None,
                     verbose=True):
    """
    builds a pandas.DataFrame from a nwbfile 
    with a given time sampling reference

    visual stimulation can be labelled either:
        - "per-protocol"   or
        - "per-protocol-and-parameters"
        - "per-protocol-and-parameters-and-timepoints"

    ---
    possibility to normalize some quantities set in the "normalize" array
    e.g. normalize = ['Running-Speed', 'Pupil-diameter', 'Whisking', 'dFoF']

    """
    data = read_NWB.Data(nwbfile, verbose=verbose)

    if subsampling is None:
        subsampling = 1

    if time_sampling_reference=='dFoF' and ('ophys' in data.nwbfile.processing):
        time = np.array(data.Fluorescence.timestamps[:])[::subsampling]
    else:
        print('taking running pseed by default')
        time = data.t_running_speed[::subsampling]

    dataframe = pandas.DataFrame({'time':time})
    dataframe.dt = time[1]-time[0] # store the time step in the metadata
    dataframe.filename = os.path.basename(nwbfile) # keep filename

    # - - - - - - - - - - - - - - - 
    # --- neural activity 
    if 'ophys' in data.nwbfile.processing:

        if not hasattr(data, 'dFoF'):
            data.build_dFoF(specific_time_sampling=time,
                            interpolation='linear',
                            verbose=verbose)

        dataframe.vNrois = data.vNrois

        for i in range(data.vNrois):

            if ('dFoF' in normalize) or (normalize=='all'):
                dataframe['dFoF-ROI%i'%i] = Normalize(data.dFoF[i,:])
            else:
                dataframe['dFoF-ROI%i'%i] = data.dFoF[i,:]

    # - - - - - - - - - - - - - - - 
    # --- behavioral characterization

    if 'Running-Speed' in data.nwbfile.acquisition:

        dataframe['Running-Speed'] = data.build_running_speed(\
                                        specific_time_sampling=time,
                                        verbose=verbose)
        if ('Running-Speed' in normalize) or (normalize=='all'):
            dataframe['Running-Speed'] = Normalize(dataframe['Running-Speed'])



    if 'Pupil' in data.nwbfile.processing:

        dataframe['Pupil-diameter'] = data.build_pupil_diameter(\
                                        specific_time_sampling=time,
                                        verbose=verbose)
        if ('Pupil-diameter') in normalize or (normalize=='all'):
            dataframe['Pupil-diameter'] = Normalize(dataframe['Pupil-diameter'])
    
    if 'Pupil' in data.nwbfile.processing:

        dataframe['Gaze-Position'] = data.build_gaze_movement(\
                                        specific_time_sampling=time,
                                        verbose=verbose)
        if ('Gaze-Position') in normalize or (normalize=='all'):
            dataframe['Gaze-Position'] = Normalize(dataframe['Gaze-Position'])
        
    if 'FaceMotion' in data.nwbfile.processing:

        dataframe['Whisking'] = data.build_facemotion(\
                                    specific_time_sampling=time,
                                        verbose=verbose)
        if ('Whisking' in normalize) or (normalize=='all'):
            dataframe['Whisking'] = Normalize(dataframe['Whisking'])
    
    # - - - - - - - - - - - - - - - 
    # --- visual stimulation

    for p, protocol in enumerate(data.protocols):

        if (protocol not in exclude_from_stim):

            episodes = process_NWB.EpisodeData(data, 
                                               protocol_id=p,
                                            verbose=verbose)

            protocol_cond = data.get_protocol_cond(p)

            if visual_stim_label=='per-protocol':

                # a binary array for this stimulation protocol,
                #       same for all stimulation parameters
                dataframe['VisStim_%s'%protocol] = \
                        build_stim_specific_array(data,
                                                  protocol_cond,
                                                  dataframe['time'],
                                normalize=('VisStim' in normalize) or (normalize=='all'))

            elif visual_stim_label=='per-protocol-and-parameters':

                # a binary array for this stimulation protocol,
                #       and a given set of stimulation parameters

                VARIED_KEYS, VARIED_VALUES, VARIED_INDICES = [], [], []
                for key in episodes.varied_parameters:
                    if key!='repeat':
                        VARIED_KEYS.append(key)
                        VARIED_VALUES.append(episodes.varied_parameters[key])
                        VARIED_INDICES.append(\
                                np.arange(len(episodes.varied_parameters[key])))
                        
                if len(VARIED_KEYS)>0:

                    for indices in itertools.product(*VARIED_INDICES):

                        # start from protocol_condition
                        episode_cond = np.zeros(len(protocol_cond), dtype=bool)
                        # then find the right parameters
                        for ep_in_protocol, in_episodes in zip(\
                                np.flatnonzero(protocol_cond),
                                episodes.find_episode_cond(VARIED_KEYS,
                                                           list(indices))):
                            # switch to True
                            if in_episodes:
                                episode_cond[ep_in_protocol] = True 


                        stim_name = 'VisStim_%s'%protocol
                        for key, index in zip(VARIED_KEYS, indices):
                            stim_name+='--%s_%s' % (key,
                                            episodes.varied_parameters[key][index])

                        dataframe[stim_name] =\
                                build_stim_specific_array(data,
                                                          episode_cond,
                                                          dataframe['time'],
                                normalize=('VisStim' in normalize) or (normalize=='all'))
                else:

                    # no varied parameter
                   dataframe['VisStim_%s'%protocol] = \
                            build_stim_specific_array(data,
                                                      protocol_cond,
                                                      dataframe['time'],
                                normalize=('VisStim' in normalize) or (normalize=='all'))

            elif visual_stim_label=='per-protocol-and-parameters-and-timepoints':

                # a binary array for this stimulation protocol,
                #       and a given set of stimulation parameters
                #       and a given frame delay from the stimulus start

                if (protocol not in exclude_from_timepoints):

                    VARIED_KEYS, VARIED_VALUES, VARIED_INDICES = [], [], []
                    for key in episodes.varied_parameters:
                        if key!='repeat':
                            VARIED_KEYS.append(key)
                            VARIED_VALUES.append(episodes.varied_parameters[key])
                            VARIED_INDICES.append(\
                                    np.arange(len(episodes.varied_parameters[key])))
                            
                    if len(VARIED_KEYS)>0:

                        for indices in itertools.product(*VARIED_INDICES):

                            # start from protocol_condition
                            episode_cond = np.zeros(len(protocol_cond), dtype=bool)
                            # then find the right parameters
                            for ep_in_protocol, in_episodes in zip(\
                                    np.flatnonzero(protocol_cond),
                                    episodes.find_episode_cond(VARIED_KEYS,
                                                               list(indices))):
                                # switch to True
                                if in_episodes:
                                    episode_cond[ep_in_protocol] = True 


                            stim_name = 'VisStim_%s'%protocol
                            for key, index in zip(VARIED_KEYS, indices):
                                stim_name+='--%s_%s' % (key,
                                                episodes.varied_parameters[key][index])
                            build_timelag_set_of_stim_specific_arrays(data, 
                                                                      dataframe,
                                                                      episode_cond, 
                                                                      stim_name=stim_name,
                                                                      pre_interval=prestim_duration,
                                                                      post_interval=poststim_duration,
                                                                      normalize=False)
                    else:

                        # no varied parameter
                        stim_name = 'VisStim_%s'%protocol
                        build_timelag_set_of_stim_specific_arrays(data, 
                                                                  dataframe,
                                                                  protocol_cond, 
                                                                  stim_name=stim_name,
                                                                  pre_interval=prestim_duration,
                                                                  post_interval=poststim_duration,
                                                                  normalize=False)

            else:
                print('visual_stim_label key not recognized !')
                print(' ---> no visual stim array in the dataframe')

   
    # adding a visual stimulation flag variable, merging all stimulation
    visualStimFlag = np.zeros(len(dataframe['time']), dtype=bool)
    for key in dataframe:
        if 'VisStim' in key:
            visualStimFlag = visualStimFlag | dataframe[key]
    dataframe['visualStimFlag'] = visualStimFlag

    return dataframe



############################################################################

def build_stim_specific_array(data, index_cond, time, 
                              normalize=False):

    array = np.zeros(len(time), dtype=bool)

    # looping over all repeats of this index
    for i in np.flatnonzero(index_cond):

        if i<data.nwbfile.stimulus['time_start_realigned'].num_samples:
            tstart = data.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = data.nwbfile.stimulus['time_stop_realigned'].data[i]

            t_cond = (time>=tstart) & (time<tstop)
            array[t_cond] = True

    # TO BE FIXED
    # if normalize:
        # return Normalize(np.array(array, dtype=float))
    # else:
        # return array
    return array


def build_timelag_set_of_stim_specific_arrays(data, DF, index_cond, 
                                              stim_name='VisStim',
                                              pre_interval=0.2,
                                              post_interval=1.8,
                                              normalize=False):


    Nframe_pre = max([1, int(pre_interval/DF.dt)]) # at least one frame
    Nframe_post = int(post_interval/DF.dt)
    Nframe_stim = int(np.min([data.nwbfile.stimulus['time_duration'].data[i]\
            for i in np.flatnonzero(index_cond)])/DF.dt)


    # set of timelag binary arrays
    for j in np.arange(-Nframe_pre, Nframe_stim+Nframe_post+1):
        DF['%s__%i' % (stim_name, j)] = np.zeros(len(DF['time']), dtype=bool)

    # looping over all repeats of this index
    for i in np.flatnonzero(index_cond):

        if i<data.nwbfile.stimulus['time_start_realigned'].num_samples:
            tstart = data.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = data.nwbfile.stimulus['time_stop_realigned'].data[i]

            iT0 = np.argmin((DF['time']-tstart)**2)
           
            for j in np.arange(-Nframe_pre, Nframe_stim+Nframe_post+1):
                DF.loc[iT0+j, '%s__%i' % (stim_name, j)] = True

    # normalize if needed
    # TO BE FIXED
    # if normalize:
        # for j in np.arange(-Nframe_pre, Nframe_stim+Nframe_post+1):
            # DF['%s__%i' % (stim_name, j)] = Normalize(np.array(DF['%s__%i' % (stim_name, j)], dtype=float))


def extract_stim_keys(dataframe,
                      indices_subset=None):
    """
    keys are of the form:

    VisStim_drifting-gratings--angle_270.0--contrast_0.33

    or in the case with timestamps:
    VisStim_drifting-gratings--angle_270.0--contrast_0.33__-2
    VisStim_drifting-gratings--angle_270.0--contrast_0.33__-1
    VisStim_drifting-gratings--angle_270.0--contrast_0.33__0
    VisStim_drifting-gratings--angle_270.0--contrast_0.33__1
                                                       [...]
    """
    STIM = {}

    for key in dataframe.keys():

        if 'VisStim_' in key:

            include = True
            if indices_subset is not None:
                # add it only if interesction with indices subset
                if not len(np.intersect1d(np.flatnonzero(dataframe[key]>0), indices_subset))>0:
                    include = False

            if include:

                s = key.replace('VisStim_', '').split('__')[0] # i.e. removing timestamps if there
                protocol = s.split('--')[0]

                if not protocol in STIM:
                    STIM[protocol] = {'DF-key':[], 'times':[]}

                STIM[protocol]['DF-key'].append(key)

                if len(key.split('__'))>1:
                    STIM[protocol]['times'].append(dataframe.dt*int(key.split('__')[1]))
                else:
                    STIM[protocol]['times'].append(0)

                keys_vals = s.split('--')
                if len(keys_vals)>1:
                    for key_val in keys_vals[1:]:
                        k, v = key_val.split('_')
                        if not k in STIM[protocol]:
                            STIM[protocol][k] = [v]
                        else:
                            STIM[protocol][k].append(v)

    # convert to numpy arrays
    for protocol in STIM:
        for key in STIM[protocol]:
            STIM[protocol][key] = np.array(STIM[protocol][key])

    return STIM




if __name__=='__main__':


    if ('.nwb' in sys.argv[-1]) and os.path.isfile(sys.argv[-1]):

        df = NWB_to_dataframe(sys.argv[-1],
                    visual_stim_label='per-protocol-and-parameters',
                                     verbose=False)

        indices = np.arange(len(df['time']))

        stim_cond = (~df['VisStim_grey-10min'])
        Nstim = int(np.sum(stim_cond)/2)

        stim_test_sets = [indices[stim_cond][Nstim:],
                           indices[stim_cond][:Nstim]]

        STIM = extract_stim_keys(df, indices_subset=stim_test_sets[0])
        print(STIM)

        # print(dataframe)
    else:

        print('you need to provide a datafile as argument')
