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

from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS

from physion.utils.compression.nwb import convert_to_nwb
from physion.utils.compression.h5 import convert_to_h5
from physion.utils.compression.binary import convert_to_binary
from physion.utils.compression.mp4 import convert_to_log8bit_mp4
from physion.utils.compression.avi import convert_to_16bit_avi

 

def imaging_to_movie_gui(self,
                       tab_id=3):

    self.source_folder = ''
    self.windows[tab_id] = 'movie conversion'

    tab = self.tabs[tab_id]
    self.cleanup_tab(tab)

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* Conversion of 2P Imaging *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel("Root Folder:", self))
    self.sourceBox = QtWidgets.QComboBox(self)
    self.sourceBox.addItems(FOLDERS)
    self.add_side_widget(tab.layout, self.sourceBox)

    self.load = QtWidgets.QPushButton('Set source folder  \u2b07', self)
    self.load.clicked.connect(self.set_source_folder)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))
    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.rm = QtWidgets.QCheckBox(' rm raw ? ', self)
    self.add_side_widget(tab.layout, self.rm)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel("Compression / Format : ", self))
    self.typeBox = QtWidgets.QComboBox()
    self.typeBox.addItems(['nwb', 'binary', '8bit-LOG-mp4', '16bit-avi (lossless)'])
    self.add_side_widget(tab.layout, self.typeBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self)
    self.gen.clicked.connect(self.run_imaging_to_movie)
    self.add_side_widget(tab.layout, self.gen)
    
    self.refresh_tab(tab)
    self.show()


def run_imaging_to_movie(self):

    Fs = find_TSeries_folders(self.source_folder)
    for f in Fs:

        if '16bit' in self.typeBox.currentText():
            print('')
            print(' [!!] Not implemented yet [!!] ')
            print('      use only from command line')
            # convert_to_avi(f)
        elif '8bit-LOG' in self.typeBox.currentText():
            convert_to_log8bit_mp4(f)
        elif 'binary' in self.typeBox.currentText():
            convert_to_binary(f)
        elif 'nwb' in self.typeBox.currentText():
            convert_to_nwb(f)
        elif 'h5' in self.typeBox.currentText():
            convert_to_h5(f)
        else:
            print(' compression type not recognized')
        print(f)

###########################


def create_compressed_folder(folder,
                             key='log8bit'):

    pathlib.Path(folder.replace('TSeries', key)).mkdir(parents=True, exist_ok=True)

    shutil.copytree(os.path.join(folder), 
                    folder.replace('TSeries', key),
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.ome.tif', #'Reference*', 
                                                  'CYCLE*', '*.bin'))

    # if os.path.isdir(\
    #         os.path.join(folder.replace('TSeries', key), 'original_suite2p')):
    #     shutil.rmtree(os.path.join(folder.replace('TSeries', key), 'original_suite2p'))

    # if os.path.isdir(os.path.join(folder.replace('TSeries', key), 'suite2p')):
    #     shutil.move(os.path.join(folder.replace('TSeries', key), 'suite2p'),
    #                 os.path.join(folder.replace('TSeries', key), 'original_suite2p'))


def find_TSeries_folders(folder):
    return [f[0] for f in os.walk(folder)\
                    if 'TSeries' in f[0].split(os.path.sep)[-1]]

def find_compressed_folders(folder, key='log8bit'):
    return [f[0] for f in os.walk(folder)\
                    if key in f[0].split(os.path.sep)[-1]]


##################  hjk

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

            if args.compression=='nwb':
                convert_to_nwb(folder)

            elif args.compression=='h5':
                convert_to_h5(folder)

            elif args.compression=='binary':
                convert_to_binary(folder)

            elif args.compression=='lossless':
                convert_to_16bit_avi(folder)

            else:
                convert_to_log8bit_mp4(folder)
            
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
