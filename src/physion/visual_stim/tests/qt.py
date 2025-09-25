from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
    QSlider, QStyle, QSizePolicy, QFileDialog
import sys, os, time
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtCore import Qt, QUrl

is_running = False

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, 
                 movie):

        self.movie= movie
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 100, 40)

        self.runBtn = QPushButton('Start/Stop')
        self.runBtn.clicked.connect(self.run)
        self.setCentralWidget(self.runBtn)

        self.show()
        
    def run(self):
        global is_running
        if not is_running:
            self.win = Window(self.movie)
            is_running = True
            self.win.play()
        else:
            self.win.close()
            is_running = False

        
# Create a QWidget-based class to represent the application window
class Window(QWidget):

    def __init__(self, movie_file):

        super().__init__()

        # Set window properties such as title, size, and icon
        self.setGeometry(200, 400, 400, 250)
        # self.showFullScreen()

        # Create a QMediaPlayer object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # Create a QVideoWidget object to display video
        videowidget = QVideoWidget()

        vboxLayout = QVBoxLayout()
        vboxLayout.setContentsMargins(0,0,0,0)
        vboxLayout.addWidget(videowidget)

        # Set the layout of the window
        self.setLayout(vboxLayout)

        # Set the video output for the media player
        self.mediaPlayer.setVideoOutput(videowidget)

        self.load_file(movie_file)

        # Display the window
        self.show()


    def play(self):
        self.mediaPlayer.play()
        # launch event loop
        self.t0 = time.time()
        self.update_dt()

    def update_dt(self):
        global is_running
        if is_running:
            if (time.time()-self.t0)>3.:
                self.mediaPlayer.setPosition(9000)
                self.t0 = time.time()
            QtCore.QTimer.singleShot(1, self.update_dt)

    def load_file(self, filename):
            self.mediaPlayer.setMedia(\
                    QMediaContent(QUrl.fromLocalFile(filename)))


# Create the application instance
app = QApplication(sys.argv)

movie = os.path.join(os.path.expanduser('~'), 'work', 'physion',
                    'src', 'physion', 'acquisition', 'protocols',
                    'movies', 'quick-spatial-mapping', 'movie.mp4')

# Create the main window instance
window = MainWindow(app, movie)

# Run the application event loop
sys.exit(app.exec_())
