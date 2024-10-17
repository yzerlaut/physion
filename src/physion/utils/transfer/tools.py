import sys, os, pathlib, shutil


def ignore_all_behav_image_folders(Dir, f):
    return ('FaceCamera' in Dir) or ('RigCamera' in Dir)

def ignore_all_tiffs(Dir, f):
    return ('.tif' in f)

def ignore_all_tiffs(Dir, f):
    return ('.tif' in f)

TYPES = {
    'Imaging (processed)':ignore_all_tiffs,
    'Imaging (processed+mp4/wmv)':{},
    'stim.+behav. (processed)':ignore_all_behav_image_folders,
    'nwb':{},
    'npy':{},
    'xml':{},
    'Imaging (+binary)':{},
    'FULL':{}
}




TYPES['Imaging (processed)']['ignore'] = ignore_all_image_folders




def ignore_all_image_folders(Dir, f):
    return ('FaceCamera' in Dir) or ('RigCamera' in Dir)


TYPES['Imaging (processed)']['include'] = \

if __name__=='__main__':

    def do_not_include(Dir, f):
        return ('FaceCamera' in Dir) or ('RigCamera' in Dir)

    def ignore_files(dir, files):
        return [f for f in files if (os.path.isfile(os.path.join(dir, f)) and\
                do_not_include(dir, f))]

    source_folder = os.path.join(os.path.expanduser('~'), 'UNPROCESSED', '2024_01_25')
    destination_folder = os.path.join(os.path.expanduser('~'), 'ASSEMBLE')

    shutil.copytree(source_folder,
                    os.path.join(destination_folder, 'copy'), 
                    ignore=ignore_files)
