import os
from PIL import Image
import numpy as np

from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser

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
