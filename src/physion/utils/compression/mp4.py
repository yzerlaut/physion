import os, sys
import cv2 as cv
from PIL import Image
import numpy as np

from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar


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
