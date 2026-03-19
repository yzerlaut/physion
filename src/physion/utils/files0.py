"""
a simple version of files, with less import
"""
import datetime, os, string, pathlib, glob

def get_date():
    return datetime.datetime.now().strftime("%Y_%m_%d")

def get_time():
    return datetime.datetime.now().strftime("%H-%M-%S")

def day_folder(root_folder):
    return os.path.join(root_folder, get_date())

def second_folder(day_folder):
    return os.path.join(day_folder, get_time())

def create_day_folder(root_folder):
    df = day_folder(root_folder)
    pathlib.Path(df).mkdir(parents=True, exist_ok=True)
    return day_folder(root_folder)

def create_second_folder(day_folder):
    pathlib.Path(second_folder(day_folder)).mkdir(parents=True, exist_ok=True)
    
def generate_datafolders(root_folder, date, time,
                           with_screen_frames_folder=False,
                           with_FaceCamera_frames_folder=False,
                           with_RigCamera_frames_folder=False,
                           with_microseconds=False):

    Day_folder = os.path.join(root_folder, date)
    date_time_folder = os.path.join(root_folder, date, time)
    
    if not os.path.exists(Day_folder):
        print('[ok] creating the folder "%s"' % Day_folder)
        pathlib.Path(Day_folder).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(date_time_folder):
        print('[ok] creating the folder "%s"' % date_time_folder)
        pathlib.Path(date_time_folder).mkdir(parents=True, exist_ok=True)

    if with_screen_frames_folder:
        pathlib.Path(os.path.join(date_time_folder,
                    'screen-frames')).mkdir(parents=True, exist_ok=True)

    if with_FaceCamera_frames_folder:
        pathlib.Path(os.path.join(date_time_folder,
                'FaceCamera-imgs')).mkdir(parents=True, exist_ok=True)
    if with_RigCamera_frames_folder:
        pathlib.Path(os.path.join(date_time_folder,
                'RigCamera-imgs')).mkdir(parents=True, exist_ok=True)

    return date_time_folder
        

def generate_filename_path(root_folder,
                           filename = '', 
                           extension='txt',
                           with_screen_frames_folder=False,
                           with_FaceCamera_frames_folder=False,
                           with_RigCamera_frames_folder=False,
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
    if with_RigCamera_frames_folder:
        pathlib.Path(os.path.join(Second_folder, 'RigCamera-imgs')).mkdir(parents=True, exist_ok=True)
        
    if not extension.startswith('.'):
        extension='.'+extension
    
    return os.path.join(Second_folder, filename+extension)


def list_dayfolder(day_folder, with_NIdaq=True):
    if with_NIdaq:
        folders = [os.path.join(day_folder, d) for d in sorted(os.listdir(day_folder)) if ((d[0] in string.digits) and (len(d)==8) and os.path.isdir(os.path.join(day_folder, d)) and os.path.isfile(os.path.join(day_folder, d, 'metadata.json')) and os.path.isfile(os.path.join(day_folder, d, 'NIdaq.npy')) and os.path.isfile(os.path.join(day_folder, d, 'NIdaq.start.npy')))]
    else:
        folders = [os.path.join(day_folder, d) for d in sorted(os.listdir(day_folder)) if ((d[0] in string.digits) and (len(d)==8) and os.path.isdir(os.path.join(day_folder, d)) and os.path.isfile(os.path.join(day_folder, d, 'metadata.json')))]
    return folders


def last_datafolder_in_dayfolder(day_folder, with_NIdaq=True):
    
    folders = list_dayfolder(day_folder, with_NIdaq=with_NIdaq)

    if len(folders)>0 and (folders[-1][-1] in string.digits):
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


if __name__=='__main__':
    import sys
