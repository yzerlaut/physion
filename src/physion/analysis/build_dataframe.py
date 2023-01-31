import os, sys, pandas, itertools
import numpy as np

from physion.analysis import read_NWB, process_NWB

def NWB_to_dataframe(nwbfile,
                     visual_stim_label='per-protocol',
                     time_sampling_reference='dFoF'):
    """
    builds a pandas.DataFrame from a nwbfile 
    with a given time sampling reference

    visual stimulation can be labelled either:
        - "per-protocol"   or
        - "per-protocol-and-parameters"

    """
    data = read_NWB.Data(nwbfile)

    if time_sampling_reference=='dFoF' and ('ophys' in data.nwbfile.processing):
        data.build_dFoF()
        time = data.t_dFoF
    else:
        print('taking running pseed by default')
        time = data.t_running_speed

    dataframe = pandas.DataFrame({'time':time})

    # - - - - - - - - - - - - - - - 
    # --- neural activity 
    if 'ophys' in data.nwbfile.processing:

        for i in range(data.nROIs):

            dataframe['dFoF-ROI%i'%i] = data.dFoF[i,:]

    # - - - - - - - - - - - - - - - 
    # --- behavioral characterization

    if 'Pupil' in data.nwbfile.processing:

        dataframe['Pupil-diameter'] = data.build_pupil_diameter(\
                                        specific_time_sampling=time)
    
    if 'Pupil' in data.nwbfile.processing:

        dataframe['Gaze-Position'] = data.build_gaze_movement(\
                                        specific_time_sampling=time)
        
    if 'FaceMotion' in data.nwbfile.processing:

        dataframe['Whisking'] = data.build_facemotion(\
                                    specific_time_sampling=time)
    
    # - - - - - - - - - - - - - - - 
    # --- visual stimulation

    for p, protocol in enumerate(data.protocols):

        episodes = process_NWB.EpisodeData(data, 
                                           protocol_id=p)


        if visual_stim_label=='per-protocol':

            # a binary array for this stimulation protocol,
            #       same for all stimulation parameters

            dataframe['VisStim-%s'%protocol] = \
                    build_stim_specific_array(data,
                                              episodes.find_episode_cond(),
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
                    episode_cond=episodes.find_episode_cond(VARIED_KEYS,
                                                            list(indices))
                    stim_name = 'VisStim-%s'%protocol
                    for key, index in zip(VARIED_KEYS, indices):
                        stim_name+='-%s-%s' % (key,
                                        episodes.varied_parameters[key][index])
                    dataframe[stim_name] =\
                            build_stim_specific_array(data,
                                                      episode_cond,
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

if __name__=='__main__':


    if ('.nwb' in sys.argv[-1]) and os.path.isfile(sys.argv[-1]):

        dataframe = NWB_to_dataframe(sys.argv[-1])
        print(dataframe)
    else:

        print('you need to provide a datafile as argument')
