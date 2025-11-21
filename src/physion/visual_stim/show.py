import os
from PyQt5 import QtWidgets, QtCore, QtMultimedia, QtMultimediaWidgets

def init_stimWindows(self, 
                    demo=False):
    
    """
     [!!] NEED TO MODIFY THIS FUNCTION WHEN SETTING UP NEW SCREENS [!!]
    """

    # we prepare the stimulus table
    self.stim.prepare_stimProps_tables(verbose=False)

    self.stimWins = []
    self.mediaPlayers, self.videowidgets = [], []

    for s in range(self.stim.screen['nScreens']):

        # Create a Qt Window
        self.stimWins.append(QtWidgets.QWidget())

        # Set window properties such as title, size, and icon
        if ('fullscreen' in self.stim.screen) and\
            self.stim.screen['fullscreen']:
            if 'Bacci-2P' in self.config['Rig']:
                self.stimWins[-1].setGeometry(-400, 400, 600, int(9./16*600))
                self.stimWins[-1].showFullScreen()
            elif 'A1-2P' in self.config['Rig']:
                self.stimWins[-1].setGeometry(2000, 400, 600, int(9./16*600))
                self.stimWins[-1].showFullScreen()
            elif 'Laptop' in self.config['Rig']:
                self.stimWins[-1].setGeometry(2000, 400, 600, int(9./16*600))
                self.stimWins[-1].showFullScreen()
            elif 'U3screens' in self.config['Rig']:
                self.stimWins[-1].setGeometry(1980+s*1024, 400, 600, int(9./16*600))
                self.stimWins[-1].showFullScreen()
        else:
            self.stimWins[-1].setGeometry(\
                    200+100*s, 400+100*s, 600, int(9./16*600))

        # Create a QMediaPlayer objects
        self.mediaPlayers.append(\
            QtMultimedia.QMediaPlayer(None, 
                    QtMultimedia.QMediaPlayer.VideoSurface))

        # Create a QVideoWidget object to display video
        self.videowidgets.append(\
            QtMultimediaWidgets.QVideoWidget())

        vboxLayout = QtWidgets.QVBoxLayout()
        vboxLayout.setContentsMargins(0,0,0,0)
        vboxLayout.addWidget(self.videowidgets[-1])

        # Set the layout of the window
        self.stimWins[s].setLayout(vboxLayout)

        # Set the video output for the media player
        self.mediaPlayers[s].setVideoOutput(\
                                    self.videowidgets[s])

        # load the movie
        if os.path.isfile(self.stim.movie_files[s]):

            self.mediaPlayers[s].setMedia(\
                    QtMultimedia.QMediaContent(\
                            QtCore.QUrl.fromLocalFile(\
                                os.path.abspath(self.stim.movie_files[s]))))

            # Set the layout of the window
            self.stimWins[s].setLayout(vboxLayout)

            self.mediaPlayers[s].play()
            self.mediaPlayers[s].pause()

            self.stimWins[s].show()

        else:

            print()
            print(' ########################################## ')
            print('    [!!]   movie file not found ! [!!]')
            print(self.stim.movie_files[s])
            print(' ########################################## ')



if __name__=='__main__':

    ######################################
    ####  visualize the stimulation   ####
    ######################################

    from physion.visual_stim.build import build_stim, get_default_params
    import json, sys, multiprocessing

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("protocol", 
                        help="""
                                either:

                                - a folder containing: 
                                        - protocol.json 
                                        - movie.mp4

                                - or simply a ".mp4" or ".wmv" movie


                             """, 
                        default='')
    parser.add_argument('-s', "--speed", type=float,
                        help="speed to visualize the stimulus (1. by default)", 
                        default=1.)
    parser.add_argument('-t', "--tstop", 
                        type=float, default=15.)
    parser.add_argument("--t0", 
                        help="start time", 
                        default=0.)
    args = parser.parse_args()

    valid = False

    if '.wmv' in args.protocol or '.mp4' in args.protocol:
        protocol = get_default_params('natural-image')
        movie_file = args.protocol
        valid = True

    else:
        Format = 'mp4' if 'linux' in sys.platform else 'wmv'
        if os.path.isfile(os.path.join(args.protocol, 'movie.%s' % Format)):
            movie_files = [os.path.join(args.protocol, 'movie.%s' % Format)]
        elif os.path.isfile(os.path.join(args.protocol, 'movie-1.%s' % Format)):
            movie_files = [\
                os.path.join(args.protocol, 'movie-%i.%s' % (i, Format)) for i in range(1,6)\
                             if os.path.isfile(os.path.join(args.protocol, 'movie-%i.%s' % (i, Format)))]
        else:
            movie_files = None
        if movie_files is not None and\
            os.path.isfile(os.path.join(args.protocol, 'protocol.json')):
            with open(os.path.join(args.protocol, 'protocol.json'), 'r') as fp:
                protocol = json.load(fp)
            protocol['demo'] = True
            valid = True
            
        if not valid:
            print('')
            print(' [!!] protocol folder not valid [!!] ')
            print('         it does not contain the protocol.json and movie.%s files' % Format)
            print('')

    if valid:

        # A minimal GUI to display the stim window 
        class MainWindow(QtWidgets.QMainWindow):

            def __init__(self):
                super(MainWindow, self).__init__()
                self.stim = build_stim(protocol)
                self.stim.movie_files = movie_files
                self.setGeometry(100, 100, 400, 40)
                self.runBtn = QtWidgets.QPushButton('Start/Stop')
                self.runBtn.clicked.connect(self.run)
                self.setCentralWidget(self.runBtn)
                self.show()
                self.runEvent = multiprocessing.Event() # to turn on/off recordings 
                
            def run(self):
                if not self.runEvent.is_set():
                    self.runEvent.set()
                    init_stimWindows(self)
                    for player in self.mediaPlayers:
                        player.play()
                else:
                    for win in self.stimWins:
                        win.close()
                    self.runEvent.clear()

        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        sys.exit(app.exec_())

