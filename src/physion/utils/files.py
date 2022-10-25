import datetime, os, string, pathlib, json, tempfile
import numpy as np

def day_folder(root_folder):
    return os.path.join(root_folder, datetime.datetime.now().strftime("%Y_%m_%d"))

def second_folder(day_folder):
    return os.path.join(day_folder, datetime.datetime.now().strftime("%H-%M-%S"))

def create_day_folder(root_folder):
    df = day_folder(root_folder)
    pathlib.Path(df).mkdir(parents=True, exist_ok=True)
    return day_folder(root_folder)

def create_second_folder(day_folder):
    pathlib.Path(second_folder(day_folder)).mkdir(parents=True, exist_ok=True)
    
def generate_filename_path(root_folder,
                           filename = '', extension='txt',
                           with_screen_frames_folder=False,
                           with_FaceCamera_frames_folder=False,
                           with_microseconds=False):

    Day_folder = day_folder(root_folder)
    Second_folder = second_folder(Day_folder)
    
    if not os.path.exists(Day_folder):
        print('creating the folder "%s"' % Day_folder)
        pathlib.Path(Day_folder).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(Second_folder):
        print('creating the folder "%s"' % Second_folder)
        pathlib.Path(Second_folder).mkdir(parents=True, exist_ok=True)

    if with_screen_frames_folder:
        pathlib.Path(os.path.join(Second_folder, 'screen-frames')).mkdir(parents=True, exist_ok=True)

    if with_FaceCamera_frames_folder:
        pathlib.Path(os.path.join(Second_folder, 'FaceCamera-imgs')).mkdir(parents=True, exist_ok=True)
        
    if not extension.startswith('.'):
        extension='.'+extension
    
    return os.path.join(Second_folder, filename+extension)


def list_dayfolder(day_folder, with_NIdaq=True):
    if with_NIdaq:
        folders = [os.path.join(day_folder, d) for d in sorted(os.listdir(day_folder)) if ((d[0] in string.digits) and (len(d)==8) and os.path.isdir(os.path.join(day_folder, d)) and os.path.isfile(os.path.join(day_folder, d, 'metadata.npy')) and os.path.isfile(os.path.join(day_folder, d, 'NIdaq.npy')) and os.path.isfile(os.path.join(day_folder, d, 'NIdaq.start.npy')))]
    else:
        folders = [os.path.join(day_folder, d) for d in sorted(os.listdir(day_folder)) if ((d[0] in string.digits) and (len(d)==8) and os.path.isdir(os.path.join(day_folder, d)) and os.path.isfile(os.path.join(day_folder, d, 'metadata.npy')))]
    return folders


def last_datafolder_in_dayfolder(day_folder, with_NIdaq=True):
    
    folders = list_dayfolder(day_folder, with_NIdaq=with_NIdaq)

    if folders[-1][-1] in string.digits:
        return folders[-1]
    else:
        print('No datafolder found, returning "./" ')
        return './'


def get_files_with_extension(folder, extension='.txt',
                             recursive=False):
    FILES = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(extension) and ('$RECYCLE.BIN' not in root):
                    FILES.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if not type(f) is str:
                f = f.decode('ascii')
            if f.endswith(extension) and ('$RECYCLE.BIN' not in folder):
                FILES.append(os.path.join(folder, f))
    return FILES


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
        FOLDERS = [f for f in next(os.walk(folder))[1] if ('TSeries' in str(f)) and (len(os.listdir(f))>frame_limit)]
    else:
        for root, subdirs, files in os.walk(folder):
            if 'TSeries' in root.split(os.path.sep)[-1] and len(files)>frame_limit:
                FOLDERS.append(os.path.join(folder, root))
            elif 'TSeries' in root.split(os.path.sep)[-1]:
                print('"%s" ignored' % root)
                print('   ----> data should be at least %i frames !' % frame_limit)
    return np.array(FOLDERS)

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
    

