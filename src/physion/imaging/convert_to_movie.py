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

    self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self)
    self.gen.clicked.connect(self.run_imaging_to_movie)
    self.add_side_widget(tab.layout, self.gen)
    
    self.refresh_tab(tab)
    self.show()

def run_imaging_to_movie(self):
    Fs = find_subfolders(self.source_folder, 'FaceCamera')
    for f in Fs:
        transform_to_movie(f, subfolder='FaceCamera', 
                           delete_raw=self.rm.isChecked())


def convert_to_16bit_avi(TS_folder,
                    delete_raw=False):

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
        print('\nBuilding the video: "%s" ' % vid_name)
        print(cmd)
        # os.system(cmd)

        if delete_raw:
            for i, f in enumerate(FILES):
                img = os.remove(os.path.join(TS_folder, f))


def convert_to_log8bit_mp4(TS_folder,
                    delete_raw=False):

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    xml_file = get_files_with_extension(TS_folder, extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        nframes = len(xml[chan]['tifFile'])
        FILES = xml[chan]['tifFile']
        movie_rate = 1./float(xml['settings']['framePeriod'])

        vid_name = os.path.join(TS_folder, 'LOG-%s.%s' %\
                                (chan.replace(' ','-'), Format))
        out = cv.VideoWriter(vid_name,
                              cv.VideoWriter_fourcc(*'mp4v'), 
                              movie_rate,
                              (Lx, Ly),
                              False)

        print('\nBuilding the video: "%s" ' % vid_name)

        success = np.zeros(len(FILES), dtype=bool)
        for i, f in enumerate(FILES):
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

            if delete_raw:
                img = os.remove(os.path.join(TS_folder, f))

        out.release()

        np.save(vid_name.replace('.%s'%Format, '-summary.npy'),
                {'compression':'log+mp4v',
                 'Frames_succesfully_in_movie':success})
        print(' [ok] done !')


def reconvert_to_tiffs_from_log8bit(vid_name):

    xml_file = get_files_with_extension(os.path.dirname(vid_name),
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    for chan in xml['channels']:

        summary = np.load(\
          vid_name.replace('.mp4','-summary.npy').replace('.wmv','-summary.npy'),
                          allow_pickle=True).item()

        cap = cv.VideoCapture(vid_name)

        nframes = len(summary['Frames_succesfully_in_movie'])
        for i, success in enumerate(\
                summary['Frames_succesfully_in_movie']):

            if success:
                # load the 8-bit frame
                ret, frame = cap.read()
                frame = np.exp(frame*np.log(2**16)/2**8)
                # convert to 16-bit
                frame = np.array(frame[:,:,0], dtype='uint16')
                im = Image.fromarray(frame)
                # write as 16bit tiff
                im.save(os.path.join(os.path.dirname(vid_name),
                                     xml[chan]['tifFile'][i]),
                                     format='TIFF')
                printProgressBar(i, nframes)

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
            

def find_subfolders(folder, cam='FaceCamera'):
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
    parser.add_argument('-d', "--delete_raw", 
                        help="protocol a json file", 
                        action="store_true")
    args = parser.parse_args()

    print('')
    for folder in find_subfolders(args.folder):

        print(' - processing', folder, ' [...]')

        if args.convert:
            if args.lossless:
                convert_to_16bit_avi(folder,
                                delete_raw=args.delete_raw)
            else:
                convert_to_log8bit_mp4(folder,
                                delete_raw=args.delete_raw)
        elif args.restore:

            xml_file = get_files_with_extension(folder,
                                                extension='.xml')[0]
            xml = bruker_xml_parser(xml_file)

            for chan in xml['channels']:
                if os.path.isfile(\
                        os.path.join(folder,
                                     'LOG-%s.mp4'%(chan.replace(' ','-')))):
                    reconvert_to_tiffs_from_log8bit(os.path.join(folder,
                                         'LOG-%s.mp4'%(chan.replace(' ','-'))))
                elif os.path.isfile(\
                        os.path.join(folder,
                                     'LOG-%s.wmv'%(chan.replace(' ','-')))):
                    reconvert_to_tiffs_from_log8bit(os.path.join(folder,
                                         'LOG-%s.wmv'%(chan.replace(' ','-'))))
                elif os.path.isfile(\
                        os.path.join(folder,
                                     '%s.avi'%(chan.replace(' ','-')))):
                    reconvert_to_tiffs_from_16bit(os.path.join(folder,
                                         '%s.avi'%(chan.replace(' ','-'))))
                else:
                    print('\n no video file to restore was found ! \n ')

        else:
            print('')
            print(10*' '+\
    ' [!!] need to choose either the "--convert" or the "--restore" option')
            print('')
