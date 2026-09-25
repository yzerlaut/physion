"""
Two compression options:

    - 1) lossless 16-bit, using ffmpeg

    - 2) convert to 8-bit mp4
        log the data to have a good resolution at low fluorescence

"""
import sys, shutil, os, pathlib, time
import cv2 as cv
from PIL import Image
import numpy as np

from PyQt5 import QtWidgets

from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS
from physion.imaging.folders import compressed_folder,\
        find_TSeries_folders, find_compressed_folders

from physion.utils.compression.nwb import convert_to_nwb
from physion.utils.compression.h5 import convert_to_h5, remove_TSeries_if_converted
from physion.utils.compression.binary import convert_to_binary
from physion.utils.compression.mp4 import convert_to_log8bit_mp4, reconvert_to_tiffs_from_log8bit
from physion.utils.compression.avi import convert_to_16bit_avi, reconvert_to_tiffs_from_16bit
from physion.gui.window import Window

# compression type (UI) -> folder key (same as in the "convert_to_..." functions)
FOLDER_KEYS = {'h5':'h5',
               'nwb':'nwb',
               'binary':'binary',
               '8bit-LOG-mp4':'log8bit',
               'log8bit':'log8bit',
               '16bit-avi (lossless)':'lossless',
               'lossless':'lossless'}




def create_compressed_folder(folder,
                             key='log8bit'):

    new_folder = compressed_folder(folder, FOLDER_KEYS.get(key, key))

    pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)

    shutil.copytree(os.path.join(folder),
                    new_folder,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.ome.tif', #'Reference*', 
                                                  'CYCLE*', '*.bin'))

    # if os.path.isdir(\
    #         os.path.join(folder.replace('TSeries', key), 'original_suite2p')):
    #     shutil.rmtree(os.path.join(folder.replace('TSeries', key), 'original_suite2p'))

    # if os.path.isdir(os.path.join(folder.replace('TSeries', key), 'suite2p')):
    #     shutil.move(os.path.join(folder.replace('TSeries', key), 'suite2p'),
    #                 os.path.join(folder.replace('TSeries', key), 'original_suite2p'))



###########################


def remove_tiff_and_binary_files(TS_folder):
    """

    we just check that the number of frames matches
    if yes:
        --> remove all tiffs and binary files !!

    """

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    xml_file = get_files_with_extension(TS_folder, 
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        nframes = len(xml[chan]['tifFile'])

        for p in np.unique(xml[chan]['depth_index']):

            vid_name = os.path.join(TS_folder, 'LOG-%s-plane%i.%s' %\
                                    (chan.replace(' ','-'), p, Format))

            cap = cv.VideoCapture(vid_name)

            nframes_vid = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            if ( (nframes-nframes_vid)/nframes ) < 0.001:
                # less than 0.1% frame difference

                print('    [!!] DELETING FOLDER IN 20s [!!] ')
                print('          (stop with Ctrl+C Ctrl+X)  ')
                print('                ', folder)
                for i in range(20):
                    printProgressBar(i, 20)
                    time.sleep(1)

                for f in os.listdir(TS_folder):
                    if f.endswith('.ome.tif')\
                            or f.endswith('.bin')\
                            or f.endswith('.env'):
                        print(f)
                        os.remove(os.path.join(TS_folder, f))


class ImagingToMovieWindow(Window):

    name = 'movie conversion'

    # functions of other modules, used as methods
    from physion.utils.transfer.gui import TransferWindow as _TransferWindow
    set_source_folder = _TransferWindow.set_source_folder

    def __init__(self, main,
                           tab_id=3):

        super().__init__(main, tab_id)
        tab = self.tab
        self.source_folder = ''


        self.add_side_widget(QtWidgets.QLabel(' _-* Conversion of 2P Imaging *-_ '))

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
        self.rm.setToolTip('h5 only: removes each "TSeries-" folder after checking that\n'
                           ' - all its tiffs are in the xml file\n'
                           ' - the h5 files match the tiffs, frame by frame (pixel-exact)\n'
                           ' - all its other files are copied to the "h5-" folder')
        self.add_side_widget(self.rm)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.add_side_widget(QtWidgets.QLabel("Compression / Format : ", self.main))
        self.typeBox = QtWidgets.QComboBox()
        self.typeBox.addItems(['h5', 'nwb', 'binary', '8bit-LOG-mp4', '16bit-avi (lossless)'])
        self.add_side_widget(self.typeBox)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self.main)
        self.gen.clicked.connect(self.run_imaging_to_movie)
        self.add_side_widget(self.gen)
    
        self.refresh_tab()
        self.show()

    def run_imaging_to_movie(self):

        Fs = find_TSeries_folders(self.source_folder)

        for f in Fs:

            create_compressed_folder(f, 
                                     self.typeBox.currentText())

            if 'avi' in self.typeBox.currentText():
                convert_to_16bit_avi(f)
            elif 'mp4' in self.typeBox.currentText():
                convert_to_log8bit_mp4(f)
            elif 'binary' in self.typeBox.currentText():
                convert_to_binary(f)
            elif 'nwb' in self.typeBox.currentText():
                convert_to_nwb(f)
            elif 'h5' in self.typeBox.currentText():
                convert_to_h5(f)
                if self.rm.isChecked():
                    # only after checking the conversion (see the h5 module)
                    removed, _ = remove_TSeries_if_converted(f)
                    self.statusBar.showMessage('"%s" %s' % (os.path.basename(f),
                            'removed' if removed else 'NOT removed (see terminal)'))
            else:
                print(' compression type not recognized')
            if self.rm.isChecked() and ('h5' not in self.typeBox.currentText()):
                print(' [!!] "rm raw" only for the h5 conversion (checked pixel-exact): %s kept' % f)
            print(f)


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("folder", 
                        default='')
    parser.add_argument("--wmv", 
                        help="protocol a json file", 
                        action="store_true")
    parser.add_argument("--compress", 
                        action="store_true")
    parser.add_argument('-c', "--compression", 
                        default='log8bit')
    parser.add_argument("--lossless", 
                        action="store_true")
    parser.add_argument("--restore", 
                        action="store_true")
    parser.add_argument("--delete", 
                        help="remove the original files", 
                        action="store_true")
    args = parser.parse_args()

    print('')

    if args.compress:

        for folder in find_TSeries_folders(args.folder):

            print(' - processing', folder, ' [...]')

            create_compressed_folder(folder, 
                                     key=args.compression)

            if 'nwb' in args.compression:
                convert_to_nwb(folder)

            elif 'h5' in args.compression:
                convert_to_h5(folder)

            elif 'binary' in args.compression:
                convert_to_binary(folder)

            elif 'avi' in args.compression:
                convert_to_16bit_avi(folder)

            elif 'mp4' in args.compression:
                convert_to_log8bit_mp4(folder)

            else:
                print("""

                compression not recognized, pick:
                    - mp4
                    - avi
                    - binary
                    - h5
                    - nwb

                """)
            
            if args.delete:
                print(' - deleting tiffs and binary in ', folder, ' [...]')
                remove_tiff_and_binary_files(folder)
                
    elif args.restore:
            
        folders  = find_compressed_folders(args.folder, 
                                          key=args.compression)
        if len(folders)>0:

            for folder in folders:

                xml_file = get_files_with_extension(folder,
                                                    extension='.xml')[0]
                xml = bruker_xml_parser(xml_file)

                for chan in xml['channels']:

                    if args.compression=='log8bit':
                        reconvert_to_tiffs_from_log8bit(folder)

                    elif args.compression=='lossless':
                        reconvert_to_tiffs_from_16bit(folder)

        else:
            print('\n no video file to restore was found ! \n ')

    else:
        print('')
        print(10*' '+\
' [!!] need to choose either the "--convert" or the "--restore" option')
        print('')
