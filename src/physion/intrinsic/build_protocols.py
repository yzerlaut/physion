import os, shutil

def to_num(direction):
    if direction=='up':
        return 1
    elif direction=='down':
        return 0
    elif direction=='left':
        return 2
    elif direction=='right':
        return 3

def build_json(direction='Up', 
               screen='Mouse-Goggles',
               bg_color=0.2,
               period=12,
               center=0,
               size=7,
               flicker_size=5,
               length=200):


    return """
{
    "Presentation": "Stimuli-Sequence",
    "Stimulus": "flickering-bar",
    "Screen": "%(screen)s",
    "units":"cm",
    "movie_refresh_freq":30.0,
    "presentation-duration": %(period)s,
    "presentation-blank-screen-color": %(bg_color)s,
    "presentation-prestim-period": 0,
    "presentation-poststim-period": 0,
    "presentation-interstim-period": 0,
    "N-repeat": 1,
    "direction": %(direction)s,
    "contrast": 0.4,
    "bar-center": %(center)s,
    "bar-length": %(length)s,
    "bar-size": %(size)s,
    "bg-color": %(bg_color)s,
    "flicker-size": %(flicker_size)s,
    "flicker-freq": 5
    } 
""" % {'direction':to_num(direction),
       'period':period,
       'bg_color':bg_color,
       'size':size,
       'flicker_size':flicker_size,
       'center':center,
       'length':length,
       'screen':screen}

if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("protocol", type=str,
                        help="""
                        either:
                            - intrinsic
                            - ocular-dominance
                        """)
    parser.add_argument("screen", type=str,
                        help="""
                        either:
                            - Dell-2020
                            - Mouse-Goggles
                            - Lilliput
                            - LN-VR-3screens
                        """)
    parser.add_argument("--bg_color", type=float, default=0.2)
    
    args = parser.parse_args()

    folder = os.path.join('physion', 'acquisition', 'protocols', 'movies', args.protocol)
    tmp_folder = os.path.join('movies-temp') # build output (removed at the end)

    from physion.visual_stim.screens import SCREENS
    nScreens = SCREENS[args.screen]['nScreens']

    def build_and_move(json_string, target_folder, basename):
        """ builds the movie and moves it as "basename.mkv" (or "basename-%i.mkv") """
        with open('temp.json', 'w') as f:
            f.write(json_string)
        os.system('python -m physion.visual_stim.build temp.json -o %s' % tmp_folder)
        os.makedirs(target_folder, exist_ok=True)
        if nScreens==1:
            os.replace(os.path.join(tmp_folder, 'temp', 'movie.mkv'),
                       os.path.join(target_folder, '%s.mkv' % basename))
        else:
            for i in range(1, nScreens+1):
                os.replace(os.path.join(tmp_folder, 'temp', 'movie-%i.mkv' % i),
                           os.path.join(target_folder, '%s-%i.mkv' % (basename, i)))

    if args.protocol=='intrinsic':
        for period in [6, 12]:
            for direction in ['up', 'down', 'left', 'right']:
                build_and_move(build_json(direction,
                                          screen=args.screen,
                                          period=period,
                                          bg_color=args.bg_color),
                               os.path.join(folder, 'flickering-bars-period%is' % period),
                               direction)

    elif args.protocol=='ocular-dominance':
        for period in [6, 12]:
            for side in ['left', 'right']:
                for direction in ['up', 'down']:
                    build_and_move(build_json(direction,
                                              screen=args.screen,
                                              period=period,
                                              center=-5 if side=='left' else 5,
                                              length=20,
                                              size=2,
                                              flicker_size=2,
                                              bg_color=args.bg_color),
                                   os.path.join(folder, 'flickering-bars-period%is' % period),
                                   '%s-%s' % (side, direction))

    if os.path.isfile('temp.json'):
        os.remove('temp.json')
    shutil.rmtree(tmp_folder, ignore_errors=True)
