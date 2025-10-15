import os
from PyQt5 import QtWidgets, QtCore, QtMultimedia, QtMultimediaWidgets

def init_stimWindow(self, 
                    demo=False):
    
    """
     [!!] NEED TO MODIFY THIS FUNCTION WHEN SETTING UP NEW SCREENS [!!]
    """
    self.stimWin = QtWidgets.QWidget()
    # we prepare the stimulus table
    self.stim.prepare_stimProps_tables(verbose=False)
    
    # Set window properties such as title, size, and icon
    if ('fullscreen' in self.stim.screen) and\
          self.stim.screen['fullscreen']:
        if 'Bacci-2P' in self.config['Rig']:
            self.stimWin.setGeometry(-400, 400, 600, int(9./16*600))
            self.stimWin.showFullScreen()
        elif 'A1-2P' in self.config['Rig']:
            self.stimWin.setGeometry(2000, 400, 600, int(9./16*600))
            self.stimWin.showFullScreen()
        elif 'Laptop' in self.config['Rig']:
            self.stimWin.setGeometry(2000, 400, 600, int(9./16*600))
            self.stimWin.showFullScreen()
    else:
        self.stimWin.setGeometry(\
                200, 400, 600, int(9./16*600))


    # Create a QMediaPlayer object
    self.mediaPlayer = QtMultimedia.QMediaPlayer(None, 
                QtMultimedia.QMediaPlayer.VideoSurface)

    # Create a QVideoWidget object to display video
    self.videowidget = QtMultimediaWidgets.QVideoWidget()

    vboxLayout = QtWidgets.QVBoxLayout()
    vboxLayout.setContentsMargins(0,0,0,0)
    vboxLayout.addWidget(self.videowidget)

    # Set the layout of the window
    self.stimWin.setLayout(vboxLayout)

    # Set the video output for the media player
    self.mediaPlayer.setVideoOutput(self.videowidget)

    # load the movie
    if os.path.isfile(self.stim.movie_file):

        self.mediaPlayer.setMedia(\
                QtMultimedia.QMediaContent(\
                        QtCore.QUrl.fromLocalFile(\
                            os.path.abspath(self.stim.movie_file))))

        # initialize the stimulation index
        self.current_index= -1 

        self.mediaPlayer.play()
        self.mediaPlayer.pause()

        self.stimWin.show()

    else:

        print()
        print(' ########################################## ')
        print('    [!!]   movie file not found ! [!!]')
        print(self.stim.movie_file)
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
        Format = 'wmv' if 'win32' in sys.platform else 'mp4'
        if os.path.isfile(os.path.join(args.protocol, 'movie.%s' % Format)) and\
            os.path.isfile(os.path.join(args.protocol, 'protocol.json')):

            movie_file = os.path.join(args.protocol, 'movie.%s' % Format)
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
                self.stim.movie_file = movie_file
                self.setGeometry(100, 100, 400, 40)
                self.runBtn = QtWidgets.QPushButton('Start/Stop')
                self.runBtn.clicked.connect(self.run)
                self.setCentralWidget(self.runBtn)
                self.show()
                self.runEvent = multiprocessing.Event() # to turn on/off recordings 
                
            def run(self):
                if not self.runEvent.is_set():
                    self.runEvent.set()
                    init_stimWindow(self)
                    self.mediaPlayer.play()
                else:
                    self.stimWin.close()
                    self.runEvent.clear()

        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        sys.exit(app.exec_())

