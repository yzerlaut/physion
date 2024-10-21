"""
Two compression options:

    - 1) lossless 16-bit, using ffmpeg

    - 2) convert to 8-bit mp4
        log the data to have a good resolution at low fluorescence

"""
import sys, shutil, os, pathlib
import cv2 as cv
from PIL import Image
import numpy as np
import ffmpeg

from PyQt5 import QtGui, QtWidgets, QtCore

from physion.assembling.tools import load_FaceCamera_data
from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS

def imaging_to_movie_gui(self,
                       tab_id=3):

    self.source_folder = ''
    self.windows[tab_id] = 'movie conversion'

    tab = self.tabs[tab_id]
    self.cleanup_tab(tab)

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* Conversion to Movie File *-_ '))

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
    self.typeBox.addItems(['8bit-LOG-mp4', '16bit-avi (lossless)'])
    self.add_side_widget(tab.layout, self.typeBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self)
    self.gen.clicked.connect(self.run_imaging_to_movie)
    self.add_side_widget(tab.layout, self.gen)
    
    self.refresh_tab(tab)
    self.show()

def run_imaging_to_movie(self):
    Fs = find_subfolders(self.source_folder)
    for f in Fs:

        if '16bit' in self.typeBox.currentText():
            print('')
            print(' [!!] Not implemented yet [!!] ')
            print('      use only from command line')
        elif '8bit-LOG' in self.typeBox.currentText():
            convert_to_log8bit_mp4(f)
        else:
            print(' compression type not recognized')
        print(f)


def convert_to_16bit_avi(TS_folder):

    xml_file = get_files_with_extension(TS_folder, extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    print('\n Analyzing: "%s" ' % TS_folder)

    for chan in xml['channels']:
   
        print('    --> Channel: ', chan)

        vid_name = os.path.join(TS_folder, '%s.avi' % chan.replace(' ','-'))

        cmd  = 'ffmpeg -i %s' % os.path.join(TS_folder,\
                    xml[chan]['tifFile'][0].replace('000001', '%06d'))+\
                    ' -c:v ffv1 '+vid_name
        print('\n  [...] Building the video: "%s" ' % vid_name)
        print(cmd)


def convert_to_log8bit_mp4(TS_folder):

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    xml_file = get_files_with_extension(TS_folder, extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        nframes = len(xml[chan]['tifFile'])
        movie_rate = 1./float(xml['settings']['framePeriod'])
        FILES = xml[chan]['tifFile']


        DICT = {'compression':'log+mp4v'}
        
        for p in np.unique(xml[chan]['depth_index']):

            vid_name = os.path.join(TS_folder, 'LOG-%s-plane%i.%s' %\
                                    (chan.replace(' ','-'), p, Format))
            out = cv.VideoWriter(vid_name,
                                  cv.VideoWriter_fourcc(*'mp4v'), 
                                  movie_rate,
                                  (Lx, Ly),
                                  False)

            print('\n  [...]  Building the video: "%s" ' % vid_name)

            plane_cond = (xml[chan]['depth_index']==p)
            success = np.zeros(len(FILES[plane_cond]), dtype=bool)
            for i, f in enumerate(FILES[plane_cond]):
                try:
                    # load 16-bit image
                    img = np.array(Image.open(os.path.join(TS_folder, f)),
                                   dtype='uint16')
                    # log and convert to 8-bit
                    img = np.array(np.log(img+1.)/np.log(2**16)*2**8, 
                                   dtype='uint8')
                    # write in movie
                    out.write(img)
                    printProgressBar(i, nframes)
                    success[i] = True
                except BaseException as be:
                    print('problem with frame:', f)

            out.release()
            print(' [ok] "%s" succesfully created !' % vid_name)
            DICT['Frames_succesfully_in_movie-plane%i'%p]= success

        np.save(os.path.join(TS_folder, 'LOG-%s-summary.npy'%chan.replace(' ','-')),
                DICT)
        print(' [ok] Frames-summary.npy succesfully created !')


def reconvert_to_tiffs_from_log8bit(TS_folder):

    xml_file = get_files_with_extension(TS_folder,
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    for chan in xml['channels']:

        summary = np.load(\
            os.path.join(TS_folder, 'LOG-%s-summary.npy'%chan.replace(' ','-')),
                          allow_pickle=True).item()

        for p in np.unique(xml[chan]['depth_index']):

            plane_cond = (xml[chan]['depth_index']==p)

            vid_name = os.path.join(TS_folder, 'LOG-%s-plane%i.%s' %\
                                    (chan.replace(' ','-'), p, Format))

            cap = cv.VideoCapture(vid_name)

            try:
                successful_frames = summary['Frames_succesfully_in_movie-plane%i'%p]
            except BaseException as be:
                # old implementation
                successful_frames = summary['Frames_succesfully_in_movie']

            nframes = len(successful_frames)
            for i, success in enumerate(successful_frames):

                if success:
                    # load the 8-bit frame
                    ret, frame = cap.read()
                    frame = np.exp(frame*np.log(2**16)/2**8)
                    # convert to 16-bit
                    frame = np.array(frame[:,:,0], dtype='uint16')
                    im = Image.fromarray(frame)
                    # write as 16bit tiff
                    im.save(os.path.join(os.path.dirname(vid_name),
                                     xml[chan]['tifFile'][plane_cond][i]),
                                     format='TIFF')
                    printProgressBar(i, nframes)
            print(' [ok] restored plane%i of "%s" ' % (p, TS_folder))

###########################


def reconvert_to_tiffs_from_16bit(vid_name):

    xml_file = get_files_with_extension(os.path.dirname(vid_name),
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    # function to extract frame:
    def extract_frame(input_vid, frame_num):
       out, _ = (
           ffmpeg
           .input(input_vid)
           .filter_('select', 'gte(n,{})'.format(frame_num))
           .output('pipe:', format='rawvideo', pix_fmt='gray16le', vframes=1)
           .run(capture_stdout=True, capture_stderr=True)
       )
       return np.frombuffer(out, np.uint16).reshape([Lx, Ly])

    for chan in xml['channels']:

        nframes = len(xml[chan]['tifFile'])

        for i in range(nframes):
            frame = extract_frame(vid_name, i)
            im = Image.fromarray(frame)
            im.save(os.path.join(os.path.dirname(vid_name),
                                 xml[chan]['tifFile'][i]),
                                 format='TIFF')
            printProgressBar(i, nframes)
            

def find_subfolders(folder):
    return [f[0] for f in os.walk(folder)\
                    if 'TSeries' in f[0].split(os.path.sep)[-1]]

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("folder", 
                        default='')
    parser.add_argument("--wmv", 
                        help="protocol a json file", 
                        action="store_true")
    parser.add_argument("--convert", 
                        action="store_true")
    parser.add_argument("--lossless", 
                        action="store_true")
    parser.add_argument("--restore", 
                        action="store_true")
    parser.add_argument("--delete", 
                        help="remove the original files", 
                        action="store_true")
    args = parser.parse_args()

    print('')
    for folder in find_subfolders(args.folder):

        print(' - processing', folder, ' [...]')

        if args.convert:
            if args.lossless:
                convert_to_16bit_avi(folder)
            else:
                convert_to_log8bit_mp4(folder)
        elif args.restore:

            xml_file = get_files_with_extension(folder,
                                                extension='.xml')[0]
            xml = bruker_xml_parser(xml_file)

            for chan in xml['channels']:
                if os.path.isfile(\
                        os.path.join(folder,
                                     'LOG-%s-summary.npy'%(chan.replace(' ','-')))):
                    reconvert_to_tiffs_from_log8bit(folder)
                elif os.path.isfile(\
                        os.path.join(folder,
                                     '%s-summary.npy'%(chan.replace(' ','-')))):
                    reconvert_to_tiffs_from_16bit(folder)
                else:
                    print('\n no video file to restore was found ! \n ')

        else:
            print('')
            print(10*' '+\
    ' [!!] need to choose either the "--convert" or the "--restore" option')
            print('')
