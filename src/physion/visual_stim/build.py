import sys
import numpy as np

import physion


def build_stim(protocol):
    """
    """
    if (protocol['Presentation']=='multiprotocol'):
        return physion.visual_stim.main.multiprotocol(protocol)
    else:
        protocol_name = protocol['Stimulus'].replace('-image','').replace('-', '_').replace('+', '_')
        try:
            return getattr(getattr(physion.visual_stim.stimuli, protocol_name), 'stim')(protocol)
        except ModuleNotFoundError:
            print('\n /!\ Protocol not recognized ! /!\ \n ')
            return None

def get_default_params(protocol_name):
    """
    """
    protocol_name = protocol_name.replace('-image','').replace('-', '_').replace('+', '_')

    try:
        Dparams = getattr(getattr(physion.visual_stim.stimuli, protocol_name), 'params')
        params = {}
        # set all params to default values
        for key in Dparams:
            params[key] = Dparams[key]
            if ' (' in key:
                new_key = key.split(' (')[0]
                params['%s-1'%new_key] = params[key]
                params['%s-2'%new_key] = params[key]
                params['N-%s'%new_key] = 0

        params['Stimulus'],  params['Presentation'] = protocol_name, ''
        params['Screen'], params['demo'] = 'Dell-2020', True
        params['seed-1'], params['seed-2'] = 1, 1
        params['N-seed'] = 1
        params['N-repeat'] = 1

        params['presentation-prestim-period'] = 0.5
        params['presentation-prestim-screen'] = 0

        params['presentation-interstim-period'] = 0.5
        params['presentation-interstim-screen'] = 0

        params['presentation-poststim-period'] = 0.5
        params['presentation-poststim-screen'] = 0

        return params

    except ModuleNotFoundError:

        print('\n /!\ Protocol not recognized ! /!\ \n ')
        return None


if __name__=='__main__':

    import os, pathlib, shutil, json

    protocol_file = sys.argv[1]

    if os.path.isfile(protocol_file) and protocol_file.endswith('.json'):

        # create the associated protocol folder in the binary folder
        protocol_folder = \
            os.path.join(os.path.dirname(protocol_file),
                'binaries',
                os.path.basename(protocol_file.replace('.json','')))
        pathlib.Path(protocol_folder).mkdir(\
                                parents=True, exist_ok=True)

        #  copy the 
        shutil.copyfile(protocol_file,
                        os.path.join(protocol_folder, 'protocol.json'))

        # build the protocol
        with open(protocol_file, 'r') as f:
            protocol = json.load(f)

        stim = build_stim(protocol)

        # loop over stims to produce the binaries and store them 
        for protocol_id in np.unique(stim.experiment['protocol_id']):
            pCond = (stim.experiment['protocol_id']==protocol_id) &\
                    (stim.experiment['repeat']==0) # taking the first one
            for stim_index in np.unique(stim.experiment['index'][pCond]):

                # get
                time_indices, frames, refresh_freq =\
                        stim.get_frames_sequence(stim_index)
                # write as binary
                np.array(frames).tofile(\
                        os.path.join(\
                            protocol_folder,\
                            'protocol-%i_index-%i.bin' % (\
                                protocol_id, stim_index)))


        # ...
        
    else:
        print('\nERROR: need to provide a valid json file as argument\n')
