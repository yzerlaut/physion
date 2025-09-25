import sys
import cv2 as cv
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
            print('\n [!!] Protocol not recognized ! [!!] \n ')
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

        params['Stimulus'],  params['Presentation'] = protocol_name, ''
        params['Screen'], params['demo'] = 'Dell-2020', True
        params['seed-1'], params['seed-2'] = 1, 1
        params['N-seed'] = 1
        params['N-repeat'] = 1

        if 'presentation-duration' not in params:
            params['presentation-duration'] = 3

        params['presentation-blank-screen-color'] = 0.5

        params['presentation-prestim-period'] = 0.5
        params['presentation-interstim-period'] = 0.5
        params['presentation-poststim-period'] = 0.5

        params['movie_refresh_freq'] = 30.
        params['units'] = 'cm'

        return params

    except ModuleNotFoundError:

        print('\n [!!] Protocol not recognized ! [!!] \n ')
        return None

class MonitoringSquare:
    """
    Used to draw a flickering square on a corner of the screen
    to monitor stimulus presentation during the experiment
    """

    def __init__(self, Stim):

        self.find_mask(Stim)

    def draw(self, image, t, tstart, tstop):
        """
        't' is the time from stimulus start !
        """
        image[self.mask] = -1. # black by default
        if (t<tstop) & (t>=tstart):
            iT = int(1000*(t-tstart))
            if (iT%1000)<150:
                image[self.mask] = 1.
            elif (iT>=500) & (iT<650):
                image[self.mask] = 1.
        return image

    def find_mask(self, Stim):
        """ find the position of the square """

        self.mask = np.zeros(Stim.screen['resolution'],
                             dtype=bool)

        S = int(Stim.screen['monitoring_square']['size'])
        X, Y = Stim.screen['resolution'] # x,y sizes

        if Stim.screen['monitoring_square']['location']=='top-right':
            self.mask[X-S:,Y-S:] = True
        elif Stim.screen['monitoring_square']['location']=='top-left':
            self.mask[X-S:,:S] = True
        elif Stim.screen['monitoring_square']['location']=='bottom-right':
            self.mask[X-S:,:S] = True
        elif Stim.screen['monitoring_square']['location']=='bottom-left':
            self.mask[:S,:S] = True
        else:
            print(30*'-'+'\n [!!]  monitoring square location not recognized !!')
            print('        --> (0,0) by default \n')
            self.mask[:S,:S] = True
        


if __name__=='__main__':

    import argparse, os, pathlib, shutil, json

    parser=argparse.ArgumentParser()
    parser.add_argument("protocol", 
                        help="protocol as a json file", 
                        default='')
    parser.add_argument("--mp4", 
                        help="force to mp4 instead of wmv", 
                        action="store_true")
    parser.add_argument('-v', "--verbose", 
                        action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.protocol) and args.protocol.endswith('.json'):

            # create the associated protocol folder in the movies folder
            protocol_folder = \
                os.path.join(os.path.dirname(args.protocol),
                    'movies',
                    os.path.basename(args.protocol.replace('.json','')))

            if os.path.isfile(os.path.join(protocol_folder, 'protocol.json')):
                # remove the previous content for security
                shutil.rmtree(protocol_folder)

            # re-create an empty one
            pathlib.Path(protocol_folder).mkdir(\
                                    parents=True, exist_ok=True)

            # build the protocol
            with open(args.protocol, 'r') as f:
                protocol = json.load(f)

            protocol['json_location'] = os.path.dirname(args.protocol)

            if args.verbose:
                protocol['verbose'] = True

            Stim = build_stim(protocol)

            #  copy the protocol infos
            with open(os.path.join(protocol_folder, 'protocol.json'), 'w') as f:
                json.dump(Stim.protocol, f, indent=4)

            def update(Stim, index):
                if index<len(Stim.experiment['index']):
                    print(' - episode %i/%i ' % (\
                            index+1, len(Stim.experiment['index'])),
                          '   protocol-id : ', 
                          Stim.experiment['protocol_id'][index])
                    if 'verbose' in protocol:
                        for k in Stim.experiment:
                            print(18*' '+'- %s:%.1f ' %\
                                    (k, Stim.experiment[k][index]))
                    tstart = Stim.experiment['time_start'][index]
                    tstop= Stim.experiment['time_stop'][index]
                    return tstart, tstop
                else:
                    return np.inf, np.inf

            # prepare the monitoring square properties
            if 'monitoring_square' in Stim.screen:
                square = MonitoringSquare(Stim)

            # prepare video file
            Format = 'mp4' if (('linux' in sys.platform) or args.mp4) else 'wmv'
            out = cv.VideoWriter(os.path.join(protocol_folder, 'movie.%s' % Format),
                                  cv.VideoWriter_fourcc(*'mp4v'), 
                                  Stim.movie_refresh_freq,
                                  Stim.screen['resolution'],
                                  False)

            # prepare the loop
            t, tend = 0, Stim.experiment['time_stop'][-1]+\
                    Stim.experiment['interstim'][-1]
            index = 0
            tstart, tstop = update(Stim, index)

            # LOOP over time to build the movie
            while t<tend:

                if t>=tstop:
                    index += 1
                    tstart, tstop = update(Stim, index)
                    
                # data in [0,1]
                if (t>=tstart) and (t<tstop):
                    data = Stim.gamma_correction(\
                            Stim.get_image(index, t-tstart))
                else:
                    data = Stim.blank_color*\
                            np.ones(Stim.screen['resolution'])

                # put the monitoring square
                if 'monitoring_square' in Stim.screen:
                    data = square.draw(data, t, tstart, tstop)

                out.write(np.array(255*np.rot90(data, k=1),
                                   dtype='uint8'))
                t+= 1./Stim.movie_refresh_freq

            np.save(os.path.join(protocol_folder, 'visual-stim.npy'), 
                    Stim.experiment)
            print('\n [ok] video file and metadata saved in: "%s" \n ' % protocol_folder)

            out.release()

    else:
        print('\nERROR: need to provide a valid json file as argument\n')
