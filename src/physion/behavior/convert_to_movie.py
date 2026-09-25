import shutil, os
from PyQt5 import QtWidgets

from physion.utils.paths import FOLDERS
from physion.utils.camera import CameraData
from physion.gui.window import Window





def find_subfolders(folder, cam='FaceCamera'):
    return [f[0].replace('%s-imgs' % cam, '')\
                for f in os.walk(folder)\
                    if f[0].split(os.path.sep)[-1]=='%s-imgs' % cam]


class CameraToMovieWindow(Window):

    name = 'movie conversion'

    # functions of other modules, used as methods
    from physion.utils.transfer.gui import TransferWindow as _TransferWindow
    set_source_folder = _TransferWindow.set_source_folder

    def __init__(self, main,
                           tab_id=3):

        self.source_folder = ''

        super().__init__(main, tab_id)
        tab = self.tab

        self.add_side_widget(QtWidgets.QLabel(' _-* Conversion to Movie File *-_ '))

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.add_side_widget(QtWidgets.QLabel("Root Folder:", self.main))
        self.sourceBox = QtWidgets.QComboBox(self.main)
        self.sourceBox.addItems(FOLDERS)
        self.add_side_widget(self.sourceBox)

        self.load = QtWidgets.QPushButton('Set source folder  \u2b07', self.main)
        self.load.clicked.connect(self.set_source_folder)
        self.add_side_widget(self.load)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))
        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.rm = QtWidgets.QCheckBox(' rm raw ? ', self.main)
        self.add_side_widget(self.rm)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self.main)
        self.gen.clicked.connect(self.convert_cameraData_to_movie
        )
        self.add_side_widget(self.gen)
    
        self.refresh_tab()
        self.show()

    def convert_cameraData_to_movie(self):
        for name in ['FaceCamera', 'RigCamera', 'ImagingCamera']:
            Fs = find_subfolders(self.source_folder, name)
            for f in Fs:
                print(name, ' :', f)
                try:
                    camData = CameraData(name, 
                                         folder=f)
                    camData.convert_to_movie()

                    # then remove if asked:
                    if self.rm.isChecked():
                        shutil.rmtree(os.path.join(f,
                                                   '%s-imgs' % self.name))
                except BaseException as be:
                    print('')
                    print(be)
                    print('')
                    print('[!!] Problem with recording,', f)
                    print('               ----> impossible to build video')
                    print('')


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("folder", 
                        default='')
    parser.add_argument("--wmv", 
                        help="protocol a json file", 
                        action="store_true")
    parser.add_argument("--delete", 
                        help="remove the original files", 
                        action="store_true")
    args = parser.parse_args()

    for name in ['FaceCamera', 'RigCamera', 'ImagingCamera']:
        for f in find_subfolders(args.folder, name):
            success = False
            try:
                camData = CameraData(name, 
                                     folder=f)
                camData.convert_to_binary()
                # camData.convert_to_movie()
                success = True
            except BaseException as be:
                print('')
                print(be)
                print('')
                print('[!!] Problem with recording,', f)
                print('               ----> impossible to build video')
                print('')

            if success and args.delete:
                print('')
                print(' [!!] removing original %s/%s-imgs/ folder' % (f, name))
                shutil.rmtree(os.path.join(f,
                                           '%s-imgs' % name))

    
