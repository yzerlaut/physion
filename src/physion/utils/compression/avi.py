import os
from PIL import Image
import numpy as np
import ffmpeg

from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar

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
            



