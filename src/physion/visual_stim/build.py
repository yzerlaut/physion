import sys
import numpy as np

import physion

def build_stim(protocol, 
               from_file=None):
    """
    """
    if from_file is not None:
        # we only build the time course of the associated video
        return physion.visual_stim.main.visual_stim(\
                                                protocol, 
                                        from_file=from_file)

    elif (protocol['Presentation']=='multiprotocol'):
        #
        return physion.visual_stim.main.multiprotocol(\
                                                protocol)

    else:
        # single protocol
        protocol_name = protocol['Stimulus'].replace('-', '_').replace('+', '_')
        try:
            return getattr(\
                        getattr(\
                            physion.visual_stim.stimuli,\
                                            protocol_name),
                                              'stim')(protocol)
        except ModuleNotFoundError:
            print('\n [!!] Protocol not recognized ! [!!] \n ')
            return None

def get_default_params(protocol_name):
    """
    """
    protocol_name = protocol_name.replace('-',
                                    '_').replace('+', '_')

    try:
        Dparams = getattr(\
                        getattr(\
                            physion.visual_stim.stimuli, protocol_name), 
                                'params')
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

#######################################################
##  ----  images of the protocols (natural images)  --- #
##  ----  and virtual saccadic explorations (VSE)   --- #
#######################################################

# name of the container of these images in the NWB files (stimulus templates)
VISUAL_STIM_IMAGES = 'visual-stim-images'


def image_key(protocol_id, image_id):
    """ key of the image "Image-ID" of protocol "protocol_id" in visual-stim.npy """
    return 'protocol_id-%i-Image-ID-%i' % (protocol_id, image_id)


def is_image_key(key):
    return key.startswith('protocol_id-') and ('-Image-ID-' in key)


def add_images_to_experiment(Stim):
    """
    stores in Stim.experiment the images of the protocols displaying images
        (stimuli with a "get_source_image" method: natural images, set of images)

        key:   "protocol_id-%i-Image-ID-%i" (see `image_key`),
                    for each "Image-ID" shown in the protocol
        value: greyscale image (uint8, 0-255), oriented as in the movie frames
                    (height x width of one screen), before contrast, background and mask
    """
    if 'Image-ID' not in Stim.experiment:
        return
    STIM = Stim.STIM if hasattr(Stim, 'STIM') else [Stim]
    protocol_ids = np.array(Stim.experiment['protocol_id'])
    image_ids = np.array(Stim.experiment['Image-ID'], dtype=object)
    for protocol_id, stim in enumerate(STIM):
        if not hasattr(stim, 'get_source_image'):
            continue
        ids = [int(i) for i in image_ids[protocol_ids==protocol_id] if i is not None]
        for image_id in np.unique(ids):
            img = np.rot90(stim.get_source_image(image_id), k=1) # as in the movie
            Stim.experiment[image_key(protocol_id, image_id)] =\
                    np.array(np.round(255*np.clip(img, 0, 1)), dtype='uint8')


def vse_key(protocol_id, seed):
    """ key of the VSE (generated with "seed") of protocol "protocol_id" in visual-stim.npy """
    return 'protocol_id-%i-VSE-%i' % (protocol_id, seed)


def is_vse_key(key):
    return key.startswith('protocol_id-') and ('-VSE-' in key)


def is_protocol_data_key(key):
    """ images and VSEs of the protocols: not episode parameters """
    return is_image_key(key) or is_vse_key(key)


def add_vse_to_experiment(Stim):
    """
    stores in Stim.experiment the virtual saccadic exploration (VSE) of the protocols having one
        (stimuli with a "vse" attribute: natural-image-VSE, grating-VSE)

        key:   "protocol_id-%i-VSE-%i" (see `vse_key`), with the seed of the VSE
                    (one VSE per protocol, generated from the "seed" of the protocol)
        value: VSE dictionary (see stimuli/virtual_scene_exploration.py)
                    't': saccade times (s, from the episode start),
                    'x', 'y': shifts after each saccade, 'max_amplitude',
                    'units': 'deg' (visual degrees) or 'pixels' (legacy protocols)
    """
    STIM = Stim.STIM if hasattr(Stim, 'STIM') else [Stim]
    for protocol_id, stim in enumerate(STIM):
        vse = getattr(stim, 'vse', None)
        if isinstance(vse, dict) and ('t' in vse):
            Stim.experiment[vse_key(protocol_id, int(stim.protocol['seed']))] =\
                    {'t':np.array(vse['t']), 'x':np.array(vse['x']), 'y':np.array(vse['y']),
                     'max_amplitude':float(vse['max_amplitude']),
                     'units':vse.get('units', 'pixels')}


class MonitoringSquare:
    """
    Used to draw a flickering square on a corner of the screen
    to monitor stimulus presentation during the experiment
    """

    def __init__(self, Stim):

        self.find_mask(Stim)

    @staticmethod
    def state(t, tstart, tstop):
        """
        square state (True=white) at time 't' for an episode in [tstart, tstop)
            white for 150ms every second, plus 150ms at 500ms
        """
        if (t<tstop) & (t>=tstart):
            iT = int(round(1000*(t-tstart), 6)) # (rounding: no float errors at the frame edges)
            if (iT%1000)<150:
                return True
            elif (iT>=500) & (iT<650):
                return True
        return False

    def draw(self, image, t, tstart, tstop):
        """
        't' is the time from stimulus start !
        """
        image[self.mask] = 1. if self.state(t, tstart, tstop) else -1. # black by default
        return image

    def find_mask(self, Stim):
        """ find the position of the square """

        self.mask = np.array(Stim.get_null_image(),
                             dtype=bool)

        S = int(Stim.screen['monitoring_square']['size'])
        X, Y = self.mask.shape

        if Stim.screen['monitoring_square']['location']=='top-right':
            self.mask[X-S:,Y-S:] = True
        elif Stim.screen['monitoring_square']['location']=='top-left':
            self.mask[X-S:,:S] = True
        elif Stim.screen['monitoring_square']['location']=='bottom-right':
            self.mask[:S,Y-S:] = True
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
    parser.add_argument('-o', "--output_folder", 
                        help="protocol as a json file", 
                        default='')
    parser.add_argument("--mp4", 
                        help="[deprecated] movies are now lossless .mkv on all platforms", 
                        action="store_true")
    parser.add_argument('-v', "--verbose", 
                        action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.protocol) and args.protocol.endswith('.json'):
            
            # create the associated protocol folder in the movies folder
            if args.output_folder!='':
                protocol_folder = \
                    os.path.join(args.output_folder,
                        os.path.basename(args.protocol.replace('.json','')))
            else:
                protocol_folder = \
                    os.path.join(
                        os.path.expanduser('~'), 'visualStim-protocols',
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
                          '   tstart: %.2f' % Stim.experiment['time_start'][index],
                          '   duration: ', 
                          Stim.experiment['time_duration'][index],
                        #   '   contrast: ', 
                        #   Stim.experiment['contrast'][index],
                          '   repeat: ', 
                          Stim.experiment['repeat'][index],
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

            # prepare the movie files (lossless, one per screen, see movie.py)
            from physion.visual_stim.movie import MovieWriter, FrameTable
            nScreens = Stim.screen['nScreens']
            out = []
            for s in range(nScreens):
                if nScreens==1:
                    filename = os.path.join(protocol_folder, 'movie.mkv')
                else:
                    filename = os.path.join(protocol_folder, 'movie-%i.mkv' % (s+1))
                print("""
                    Screen %i, %s""" % (s+1, filename))
                out.append(MovieWriter(filename,
                                       Stim.screen['resolution'],
                                       Stim.movie_refresh_freq))
            table = FrameTable(Stim.movie_refresh_freq,
                               Stim.screen['resolution'])
            print()

            # prepare the loop
            tend = Stim.experiment['time_stop'][-1]+\
                    Stim.experiment['interstim'][-1]
            nFrames = int(np.ceil(tend*Stim.movie_refresh_freq-1e-9))
            index = 0
            tstart, tstop = update(Stim, index)

            # LOOP over frames to build the movie
            #    (frame times as i/fps: no accumulation of rounding errors)
            for iFrame in range(nFrames):

                t = iFrame/Stim.movie_refresh_freq

                while t>=tstop:
                    index += 1
                    tstart, tstop = update(Stim, index)

                # data in [0,1]
                # GAMMA CORRECTION ONLY AT THAT POINT !
                if (t>=tstart) and (t<tstop):
                    data = Stim.gamma_correction(\
                            Stim.get_image(index, t-tstart))
                    episode = index
                else:
                    data = Stim.gamma_correction(\
                                Stim.blank_color+\
                                    Stim.get_null_image())
                    episode = -1

                # put the monitoring square
                square_state = None
                if 'monitoring_square' in Stim.screen:
                    data = square.draw(data, t, tstart, tstop)
                    square_state = square.state(t, tstart, tstop)

                frames = []
                for s, o in enumerate(out):
                    dataS = Stim.restrict_to_screen(data, 
                                                    screen_id=s+1)
                    # [0,1] -> [0,255] (truncation as before, but clipped: no uint8 overflow)
                    frame = np.clip(np.nan_to_num(255*np.rot90(dataS, k=1)),
                                    0, 255).astype(np.uint8)
                    o.write(frame)
                    frames.append(frame)
                table.add(frames, episode, square_state)

            for o in out:
                o.close()
            table.save(protocol_folder)

            # the images displayed (natural images) -> "protocol_id-%i-Image-ID-%i" keys
            add_images_to_experiment(Stim)
            # the virtual saccadic explorations -> "protocol_id-%i-VSE-%i" keys
            add_vse_to_experiment(Stim)

            np.save(os.path.join(protocol_folder, 'visual-stim.npy'),
                    Stim.experiment)

            print('\n [ok] video file and metadata saved in: "%s" \n ' % protocol_folder)

    else:

        print('\nERROR: need to provide a valid json file as argument\n')
