import os, sys, itertools, pandas
import numpy as np

from physion.analysis import read_NWB, process_NWB

def NWB_to_dataframe(nwbfile,
                     visual_stim_label='per-protocol',
                     time_sampling_reference='dFoF',
                     verbose=True):
    """
    builds a pandas.DataFrame from a nwbfile 
    with a given time sampling reference

    visual stimulation can be labelled either:
        - "per-protocol"   or
        - "per-protocol-and-parameters"

    """
    data = read_NWB.Data(nwbfile, verbose=verbose)

    if time_sampling_reference=='dFoF' and ('ophys' in data.nwbfile.processing):
        data.build_dFoF(verbose=verbose)
        time = data.t_dFoF
    else:
        print('taking running pseed by default')
        time = data.t_running_speed

    dataframe = pandas.DataFrame({'time':time})

    # - - - - - - - - - - - - - - - 
    # --- neural activity 
    if 'ophys' in data.nwbfile.processing:

        for i in range(data.vNrois):

            dataframe['dFoF-ROI%i'%i] = data.dFoF[i,:]

    # - - - - - - - - - - - - - - - 
    # --- behavioral characterization

    if 'Running-Speed' in data.nwbfile.acquisition:

        dataframe['Running-Speed'] = data.build_running_speed(\
                                        specific_time_sampling=time,
                                        verbose=verbose)


    if 'Pupil' in data.nwbfile.processing:

        dataframe['Pupil-diameter'] = data.build_pupil_diameter(\
                                        specific_time_sampling=time,
                                        verbose=verbose)
    
    if 'Pupil' in data.nwbfile.processing:

        dataframe['Gaze-Position'] = data.build_gaze_movement(\
                                        specific_time_sampling=time,
                                        verbose=verbose)
        
    if 'FaceMotion' in data.nwbfile.processing:

        dataframe['Whisking'] = data.build_facemotion(\
                                    specific_time_sampling=time,
                                        verbose=verbose)
    
    # - - - - - - - - - - - - - - - 
    # --- visual stimulation

    for p, protocol in enumerate(data.protocols):

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
                                              dataframe['time'])  
        else:

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
                                                      dataframe['time'])  
            else:

                # no varied parameter
                dataframe['VisStim-%s'%protocol] = \
                        build_stim_specific_array(data,
                                                  protocol_cond,
                                                  dataframe['time'])  

    return dataframe

def build_stim_specific_array(data, index_cond, time):

    array = 0*time

    # looping over all repeats of this index
    for i in np.flatnonzero(index_cond):

        if i<data.nwbfile.stimulus['time_start_realigned'].num_samples:
            tstart = data.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = data.nwbfile.stimulus['time_stop_realigned'].data[i]

            t_cond = (time>=tstart) & (time<tstop)
            array[t_cond] = 1

    return array

def extract_stim_keys(dataframe):

    STIM = {}
    for key in dataframe.keys():
        if 'VisStim_' in key:
            s = key.replace('VisStim_', '')
            protocol = s.split('--')[0]

            if not protocol in STIM:
                STIM[protocol] = {'DF-key':[]}

            STIM[protocol]['DF-key'].append(key)

            keys_vals = s.split('--')
            if len(keys_vals)>1:
                for key_val in keys_vals[1:]:
                    k, v = key_val.split('_')
                    if not k in STIM[protocol]:
                        STIM[protocol][k] = [v]
                    else:
                        STIM[protocol][k].append(v)


    return STIM

if __name__=='__main__':


    if ('.nwb' in sys.argv[-1]) and os.path.isfile(sys.argv[-1]):

        dataframe = NWB_to_dataframe(sys.argv[-1],
                    visual_stim_label='per-protocol-and-parameters',
                                     verbose=False)

        print(extract_stim_keys(dataframe))
        # print(dataframe)
    else:

        print('you need to provide a datafile as argument')
