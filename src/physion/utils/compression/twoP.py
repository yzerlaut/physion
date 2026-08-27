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
import ffmpeg

from PyQt5 import QtWidgets

from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS

from pynwb import NWBHDF5IO, NWBFile
from dateutil.tz import tzlocal
from datetime import datetime
from uuid import uuid4
from hdmf.data_utils import DataChunkIterator

from pynwb.ophys import (
    CorrectedImageStack,
    Fluorescence,
    ImageSegmentation,
    MotionCorrection,
    OnePhotonSeries,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)


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


def write_nwb(filename, 
              data=np.zeros((512,512,10), dtype=np.uint16),
              close_io=True):

    start_time = datetime(2017, 4, 3, 11, tzinfo=tzlocal())
    nwbfile = NWBFile(
        session_description="demonstrate iterative write",
        identifier=str(uuid4()),
        session_start_time=start_time,
    )

    device_model = nwbfile.create_device_model(
        name="Thorlabs Bergamo II Model",
        description="Two-photon microscope for in vivo imaging",
        manufacturer="Thorlabs",
        model_number="Bergamo II",
    )

    device = nwbfile.create_device(
        name="Thorlabs Bergamo II",
        description="Two-photon microscope for in vivo imaging",
        model=device_model,
        serial_number="SN-123456789",
    )

    optical_channel = OpticalChannel(
        name="OpticalChannel",
        description="an optical channel",
        emission_lambda=500.0,
    )

    imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane",
        optical_channel=optical_channel,
        imaging_rate=30.0,
        description="a very interesting part of the brain",
        device=device,
        excitation_lambda=600.0,
        indicator="GFP",
        location="V1",
        grid_spacing=[0.01, 0.01],
        grid_spacing_unit="meters",
        origin_coords=[1.0, 2.0, 3.0],
        origin_coords_unit="meters",
    )

    # the image data will be stored inside the NWB file
    two_p_series = TwoPhotonSeries(
        name="TwoPhotonSeries",
        description="Raw 2p data",
        data=data,
        imaging_plane=imaging_plane,
        rate=1.0,
        unit="normalized amplitude",
    )
    nwbfile.add_acquisition(two_p_series)

    # Write the data to file
    io = NWBHDF5IO(filename, "w")
    io.write(nwbfile)
    if close_io:
        io.close()
        del io
        io = None
    return io

def convert_to_nwb_Movie(TS_folder):

    xml_file = get_files_with_extension(TS_folder, 
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    if not os.path.isdir(os.path.join(TS_folder.replace('TSeries', 'nwb'))):
        os.mkdir(os.path.join(TS_folder.replace('TSeries', 'nwb')))

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        nframes = len(xml[chan]['tifFile'])
        movie_rate = 1./float(xml['settings']['framePeriod'])
        FILES = xml[chan]['tifFile']

        DICT = {'compression':'None'}
        
        for p in np.unique(xml[chan]['depth_index']):

            plane_cond = (xml[chan]['depth_index']==p)

            vid_name = os.path.join(TS_folder.replace('TSeries', 'nwb'),
                                     '%s-plane%i.nwb' %\
                                    (chan.replace(' ','-'), p))

            def data_iter_func(max_chunks=10):
                i = 0
                while i<len(FILES[plane_cond]):
                    img = np.array(Image.open(os.path.join(TS_folder, FILES[plane_cond][i])),
                                dtype='uint16')
                    print(img)
                    i+=1
                    yield img
                return

            write_nwb(vid_name,
                      data = DataChunkIterator(data=data_iter_func,
                                               dtype=np.dtype('uint16')))

            # print('\n  [...]  Building the binary file: "%s" ' % vid_name)
            # with open(vid_name, 'wb') as bin:

            #     for i, f in enumerate(FILES[plane_cond]):
            #         try:
            #             # load 16-bit image
            #             img = np.array(Image.open(os.path.join(TS_folder, f)),
            #                         dtype='uint16')
            #             # write in movie
            #             bin.write(img.tobytes())
            #             if i%100==0:
            #                 printProgressBar(i, nframes)
            #             success[i] = True
            #         except BaseException as be:
            #             print('problem with frame:', f)
            #             print(be)

            print(' [ok] "%s" succesfully created !' % vid_name)
            # DICT['Frames_succesfully_in_movie-plane%i'%p]= success

        np.save(os.path.join(TS_folder.replace('TSeries', 'nwb'), 
                             '%s-summary.npy'%chan.replace(' ','-')),
                DICT)
        print(' [ok] Frames-summary.npy succesfully created !')

def convert_to_binary(TS_folder):

    xml_file = get_files_with_extension(TS_folder, 
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    if not os.path.isdir(os.path.join(TS_folder.replace('TSeries', 'binary'))):
        os.mkdir(os.path.join(TS_folder.replace('TSeries', 'binary')))

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        nframes = len(xml[chan]['tifFile'])
        movie_rate = 1./float(xml['settings']['framePeriod'])
        FILES = xml[chan]['tifFile']

        DICT = {'compression':'log+mp4v'}
        
        for p in np.unique(xml[chan]['depth_index']):

            vid_name = os.path.join(TS_folder.replace('TSeries', 'binary'),
                                     '%s-plane%i.bin' %\
                                    (chan.replace(' ','-'), p))

            plane_cond = (xml[chan]['depth_index']==p)
            success = np.zeros(len(FILES[plane_cond]), dtype=bool)

            print('\n  [...]  Building the binary file: "%s" ' % vid_name)
            with open(vid_name, 'wb') as bin:

                for i, f in enumerate(FILES[plane_cond]):
                    try:
                        # load 16-bit image
                        img = np.array(Image.open(os.path.join(TS_folder, f)),
                                    dtype='uint16')
                        # write in movie
                        bin.write(img.tobytes())
                        if i%100==0:
                            printProgressBar(i, nframes)
                        success[i] = True
                    except BaseException as be:
                        print('problem with frame:', f)
                        print(be)

            print(' [ok] "%s" succesfully created !' % vid_name)
            DICT['Frames_succesfully_in_movie-plane%i'%p]= success

        np.save(os.path.join(TS_folder.replace('TSeries', 'binary'), 
                             '%s-summary.npy'%chan.replace(' ','-')),
                DICT)
        print(' [ok] Frames-summary.npy succesfully created !')


def run_imaging_to_movie(self):

    Fs = find_TSeries_folders(self.source_folder)
    for f in Fs:

        if '16bit' in self.typeBox.currentText():
            print('')
            print(' [!!] Not implemented yet [!!] ')
            print('      use only from command line')
        elif '8bit-LOG' in self.typeBox.currentText():
            convert_to_log8bit_mp4(f)
        elif 'binary' in self.typeBox.currentText():
            convert_to_binary(f)
        elif 'nwb' in self.typeBox.currentText():
            convert_to_nwb_Movie(f)
        else:
            print(' compression type not recognized')
        print(f)


def convert_to_16bit_avi(TS_folder):

    xml_file = get_files_with_extension(TS_folder, extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    print('\n Analyzing: "%s" ' % TS_folder)

    if not os.path.isdir(os.path.join(TS_folder.replace('TSeries', 'lossless'))):
        os.mkdir(os.path.join(TS_folder.replace('TSeries', 'lossless')))

    for chan in xml['channels']:
   
        print('    --> Channel: ', chan)

        vid_name = os.path.join(TS_folder.replace('TSeries', 'lossless'),
                                '%s.avi' % chan.replace(' ','-'))

        cmd  = 'ffmpeg -i %s' % os.path.join(TS_folder,\
                    xml[chan]['tifFile'][0].replace('000001', '%06d'))+\
                    ' -c:v ffv1 '+vid_name
        print('\n  [...] Building the video: "%s" ' % vid_name)
        print()
        print('  command to execute: ')
        print(cmd)
        print()



def convert_to_log8bit_mp4(TS_folder):

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    xml_file = get_files_with_extension(TS_folder, 
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])


    if not os.path.isdir(os.path.join(TS_folder.replace('TSeries', 'log8bit'))):
        os.mkdir(os.path.join(TS_folder.replace('TSeries', 'log8bit')))

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        nframes = len(xml[chan]['tifFile'])
        movie_rate = 1./float(xml['settings']['framePeriod'])
        FILES = xml[chan]['tifFile']


        DICT = {'compression':'log+mp4v'}
        
        for p in np.unique(xml[chan]['depth_index']):

            vid_name = os.path.join(TS_folder.replace('TSeries', 'log8bit'),
                                     '%s-plane%i.%s' %\
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
                    img = np.array(np.log(img+1.)/np.log(2**16)*(2**8-1), 
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

        np.save(os.path.join(TS_folder.replace('TSeries', 'log8bit'), 
                             '%s-summary.npy'%chan.replace(' ','-')),
                DICT)
        print(' [ok] Frames-summary.npy succesfully created !')


def reconvert_to_tiffs_from_log8bit(TS_folder):

    xml_file = get_files_with_extension(TS_folder,
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    for chan in xml['channels']:

        summary = np.load(\
            os.path.join(TS_folder, '%s-summary.npy'%chan.replace(' ','-')),
                          allow_pickle=True).item()

        for p in np.unique(xml[chan]['depth_index']):

            plane_cond = (xml[chan]['depth_index']==p)

            vid_name = os.path.join(TS_folder, '%s-plane%i.%s' %\
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
                    frame = np.exp(frame*np.log(2**16-1)/(2**8-1))-1
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
                convert_to_nwb_Movie(folder)

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
