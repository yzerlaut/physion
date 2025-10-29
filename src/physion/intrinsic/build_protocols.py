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
                        """)
    parser.add_argument("--bg_color", type=float, default=0.2)
    
    args = parser.parse_args()

    folder = os.path.join('physion', 'acquisition', 'protocols', 'movies', args.protocol)

    if args.protocol=='intrinsic':
        for period in [6, 12]:
            if not os.path.isdir(folder):
                os.mkdir(folder)
            for direction in ['up', 'down', 'left', 'right']:
                # create the directory
                if not os.path.isdir(os.path.join(folder, 'flickering-bars-period%is' % period)):
                    os.mkdir(os.path.join(folder, 'flickering-bars-period%is' % period))
                # write the json
                with open('temp.json', 'w') as f:
                    f.write(build_json(direction, 
                                       screen=args.screen,
                                       period=period, 
                                       bg_color=args.bg_color))
                # build the movie
                os.system('python -m physion.visual_stim.build temp.json')
                os.rename(os.path.join('movies', 'temp', 'movie.wmv'),
                          os.path.join(folder,
                                       'flickering-bars-period%is' % period,
                                       '%s.wmv' % direction))

    elif args.protocol=='ocular-dominance':
        for period in [6, 12]:
            if not os.path.isdir(folder):
                os.mkdir(folder)
            for side in ['left', 'right']:
                for direction in ['up', 'down']:
                    # create the directory
                    if not os.path.isdir(os.path.join(folder, 'flickering-bars-period%is' % period)):
                        os.mkdir(os.path.join(folder, 'flickering-bars-period%is' % period))
                    # write the json
                    with open('temp.json', 'w') as f:
                        f.write(build_json(direction, 
                                           screen=args.screen,
                                           period=period, 
                                           center=-5 if side=='left' else 5,
                                           length=20,
                                           size=2,
                                           flicker_size=2,
                                           bg_color=args.bg_color))
                    # build the movie
                    os.system('python -m physion.visual_stim.build temp.json')
                    os.rename(os.path.join('movies', 'temp', 'movie.wmv'),
                          os.path.join(folder,
                                       'flickering-bars-period%is' % period,
                                       '%s-%s.wmv' % (side, direction)))

        os.remove('temp.json')
    shutil.rmtree('movies')
