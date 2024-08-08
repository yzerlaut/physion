import sys
import cv2
import numpy as np

import physion


def build_stim(protocol):
    """
    """
    if (protocol['Presentation']=='multiprotocol'):
        return physion.visual_stim.main.multiprotocol(protocol)
    else:
        protocol_name = protocol['Stimulus'].replace('-', '_').replace('+', '_')
        try:
            return getattr(getattr(physion.visual_stim.stimuli,\
                                protocol_name), 'stim')(protocol)
        except ModuleNotFoundError:
            print('\n /!\ Protocol not recognized ! /!\ \n ')
            return None

def get_default_params(protocol_name):
    """
    """
    protocol_name = protocol_name.replace('-', '_').replace('+', '_')

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

        params['presentation-blank-screen-color'] = 0

        params['presentation-prestim-period'] = 0.5
        params['presentation-interstim-period'] = 0.5
        params['presentation-poststim-period'] = 0.5

        return params

    except ModuleNotFoundError:

        print('\n /!\ Protocol not recognized ! /!\ \n ')
        return None

def write_binary(stim, index, protocol_id):

    time_indices, frames, refresh_freq = stim.get_frames_sequence(index)
    print('writing: protocol-%i_index-%i.bin' % (protocol_id, index))
    Frames = np.array([255*(1.+f.T)/2. for f in frames], dtype=np.uint8)
    # write as binary
    Frames.tofile(os.path.join(protocol_folder,\
                    'protocol-%i_index-%i.bin' % (protocol_id, index)))
    # write as npy 
    np.save(os.path.join(\
                protocol_folder,\
                'protocol-%i_index-%i.npy' % (protocol_id, index)),
                {'refresh_freq':refresh_freq,
                 'time_indices':time_indices,
                 'binary_shape': Frames.shape})


if __name__=='__main__':

    import argparse, os, pathlib, shutil, json

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("protocol", 
                        help="protocol a json file", 
                        default='')
    parser.add_argument("--fps", type=int,
                        help="Frame Per Seconds in the mp4 movie",
                        default=30)
    args = parser.parse_args()

    if os.path.isfile(args.protocol) and args.protocol.endswith('.json'):

        movie_filename = os.path.join(os.path.dirname(args.protocol),
                                'movies',
                        os.path.basename(args.protocol.replace('.json','.mp4')))
        ok = False
        if os.path.isfile(movie_filename):
            if (input(' /!\  "%s" already exists\n       replace ? [no]/yes   ' % movie_filename) in ['y', 'yes']):
                os.remove(movie_filename)
                ok = True
        else:
            ok = True

        if ok:

            # build the protocol
            with open(args.protocol, 'r') as f:
                protocol = json.load(f)

            protocol['no-window'] = True

            Stim = build_stim(protocol)

            def update(Stim, index):
                if index<len(Stim.experiment['index']):
                    print(' - episode %i/%i ' % (\
                            index+1, len(Stim.experiment['index'])))
                    tstart = Stim.experiment['time_start'][index]
                    tstop= Stim.experiment['time_stop'][index]
                    return tstart, tstop
                else:
                    return np.inf, np.inf

            # prepare video file
            out = cv2.VideoWriter(movie_filename,
                                  cv2.VideoWriter_fourcc(*'mp4v'), 
                                  args.fps, 
                                  Stim.screen['resolution'],
                                  False)

            # prepare the loop
            t, tend = 0, Stim.experiment['time_stop'][-1]+\
                    Stim.experiment['interstim'][-1]
            index = 0
            tstart, tstop = update(Stim, index)
            print('')

            # LOOP over time to build the movie
            while t<tend:

                if t>=tstop:
                    index += 1
                    tstart, tstop = update(Stim, index)

                if (t>=tstart) and (t<tstop):
                    data = (Stim.get_image(index, t-tstart)+1.)/2.
                else:
                    data = (Stim.blank_color*np.ones(\
                                Stim.screen['resolution'])+1.)/2.

                out.write(np.array(256*np.rot90(data, k=1), 
                                   dtype='uint8'))
                t+= 1./args.fps

            print('\n [ok] video file saved as: "%s" \n ' % movie_filename)
            out.release()

    else:
        print('\nERROR: need to provide a valid json file as argument\n')
