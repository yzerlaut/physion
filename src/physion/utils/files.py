import datetime, os, string, pathlib, glob
import numpy as np

from .files0 import * #


def get_files_with_given_exts(dir='./', EXTS=['npz','abf','bin']):
    """  DEPRECATED, use the function above !!"""
    FILES = []
    for ext in EXTS:
        for file in os.listdir(dir):
            if file.endswith(ext):
                FILES.append(os.path.join(dir, file))
    return np.array(FILES)

def get_TSeries_folders(folder, frame_limit=-1, limit_to_subdirectories=False):
    
    """ get files of a given extension and sort them..."""
    FOLDERS = []
    if limit_to_subdirectories:
        FOLDERS = [f for f in next(os.walk(folder))[1]\
                    if (\
                        ('TSeries' in str(f)) or \
                        ('log8bit-' in str(f)) or \
                        ('lossless-' in str(f)))\
                          and (len(os.listdir(f))>frame_limit)]
    else:
        for root, subdirs, files in os.walk(folder):
            if (len(files)>frame_limit) and\
                (\
                    ('TSeries' in str(root.split(os.path.sep)[-1])) or \
                    ('log8bit-' in str(root.split(os.path.sep)[-1])) or \
                    ('lossless-' in str(root.split(os.path.sep)[-1]))\
                    ):
                FOLDERS.append(os.path.join(folder, root))
            elif 'TSeries' in root.split(os.path.sep)[-1]:
                print('"%s" ignored' % root)
                print('   ----> data should be at least %i frames !' % frame_limit)
    # print(np.sort(np.array(FOLDERS))
    return np.sort(np.array(FOLDERS))

def insure_ordered_frame_names(df):
    # insuring nice order of screen frames
    filenames = os.listdir(os.path.join(df,'screen-frames'))
    if len(filenames)>0:
        nmax = np.max(np.array([len(fn) for fn in filenames]))
        for fn in filenames:
            n0 = len(fn)
            if n0<nmax:
                os.rename(os.path.join(df,'screen-frames', fn),
                          os.path.join(df,'screen-frames', fn.replace('frame', 'frame'+'0'*(nmax-n0))))

def insure_ordered_FaceCamera_picture_names(df):
    # insuring nice order of screen frames
    filenames = os.listdir(os.path.join(df,'FaceCamera-imgs'))
    if len(filenames)>0:
        nmax = np.max(np.array([len(fn) for fn in filenames]))
        for fn in filenames:
            n0 = len(fn)
            if n0<nmax:
                os.rename(os.path.join(df,'FaceCamera-imgs',fn),
                          os.path.join(df,'FaceCamera-imgs','0'*(nmax-n0)+fn))
                

def from_folder_to_datetime(folder):

    s = folder.split(os.path.sep)[-2:]

    try:
        date = s[0].split('_')
        return date[2]+'/'+date[1]+'/'+date[0], s[1].replace('-', ':')
    except Exception:
        return '', folder

def folderName_to_daySeconds(datafolder):

    Hour = int(datafolder.split('-')[0][-2:])
    Min = int(datafolder.split('-')[1])
    Seconds = int(datafolder.split('-')[2][:2])

    return 60.*60.*Hour+60.*Min+Seconds
            
def computerTimestamp_to_daySeconds(t):

    s = str(datetime.timedelta(seconds=t))

    Hour = int(s.split(':')[0][-2:])
    Min = int(s.split(':')[1])
    Seconds = float(s.split(':')[2])
    
    return 60*60*Hour+60*Min+Seconds

def get_latest_file(folder):
    list_of_files = glob.glob(os.path.join(folder, '*')) 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

if __name__=='__main__':
    import sys
    print(get_latest_file(sys.argv[-1]))
