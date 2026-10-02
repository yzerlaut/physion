import numpy as np
from physion.acquisition.tools import find_line_props


def TTL_signal(visual_stim, 
               metadata,
               stim_data):

    pre_duration = stim_data['pre_window']
    post_duration = stim_data['post_window']

    props = find_line_props(\
        metadata['NIdaq']['digital-outputs']['line-labels'],
                            'LED-optogenetics-activation')

    def step(tStart, tStop):
        return {'channel':props['chan'], 
                'onset':tStart-pre_duration, 
                'duration':tStop-tStart+post_duration+pre_duration}

    digital_output_steps = []

    for tStart, tStop, repeat, protocol_id in zip(
                    visual_stim.experiment['time_start'],
                    visual_stim.experiment['time_stop'],
                    visual_stim.experiment['repeat'],
                    visual_stim.experiment['protocol_id'],
                ):
        if 'Protocol-%i' % (protocol_id+1) in stim_data:
            if stim_data['Protocol-%i' % (protocol_id+1)]['trials']=='all':
                digital_output_steps.append(step(tStart, tStop))
            elif (repeat%2==0) and stim_data['Protocol-%i' % (protocol_id+1)]['trials']=='even':
                digital_output_steps.append(step(tStart, tStop))
            elif (repeat%2==1) and stim_data['Protocol-%i' % (protocol_id+1)]['trials']=='odd':
                digital_output_steps.append(step(tStart, tStop))

    return digital_output_steps