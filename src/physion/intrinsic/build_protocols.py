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
               period=12):


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
    "bar-size": 7,
    "bg-color": %(bg_color)s,
    "flicker-size": 5,
    "flicker-freq": 5
    } 
""" % {'direction':to_num(direction),
       'period':period,
       'bg_color':bg_color,
       'screen':screen}

if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("screen", type=str,
                        help="""
                        either:
                            - Dell-2020
                            - Mouse-Goggles
                        """)
    parser.add_argument("--bg_color", type=float, default=0.2)
    
    args = parser.parse_args()

    for period in [6, 12]:
        if not os.path.isdir('physion/acquisition/protocols/movies/intrinsic/'):
            os.mkdir('physion/acquisition/protocols/movies/intrinsic')
        for direction in ['up', 'down', 'left', 'right']:
            # create the directory
            if not os.path.isdir('physion/acquisition/protocols/movies/intrinsic/flickering-bars-period%is' % period):
                os.mkdir('physion/acquisition/protocols/movies/intrinsic/flickering-bars-period%is' % period)
            # write the json
            with open('temp.json', 'w') as f:
                f.write(build_json(direction, 
                                   screen=args.screen,
                                   period=period, 
                                   bg_color=args.bg_color))
            # build the movie
            os.system('python -m physion.visual_stim.build temp.json --wmv')
            os.rename('movies/temp/movie.wmv', 
    'physion/acquisition/protocols/movies/intrinsic/flickering-bars-period%is/%s.wmv' % (period, direction))

        os.remove('temp.json')
    shutil.rmtree('movies')
