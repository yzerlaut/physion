"""
script to build the different ignore function to be used in shutil.copytree
    for COPYING SPECIFIC SUBSETS OF DATA


by default shutil doesn't have an "include_pattern" function,
    so taken from:
https://stackoverflow.com/questions/52071642/python-copying-the-files-with-include-pattern

"""
import shutil

from fnmatch import fnmatch, filter
from os.path import isdir, join

def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns

TYPES = {
    '':None,
    'processed-Imaging':shutil.ignore_patterns('*.ome.tif', 
                                               'Reference*', 
                                               'CYCLE*',
                                               '*.bin',
                                               '*.mp4', '*.avi',
                                               'FaceCamera-*', 'RigCamera-*',
                                               '*.env'),
    'processed-Imaging-wVids':shutil.ignore_patterns('*.ome.tif', 
                                                    'Reference*', 
                                                    'CYCLE*',
                                                    '*.bin',
                                                    'FaceCamera-*', 'RigCamera-*',
                                                    '*.env'),
    'processed-Imaging-wBinary':shutil.ignore_patterns('*.ome.tif', 
                                                       'Reference*', 
                                                       'FaceCamera-*', 'RigCamera-*',
                                                       '*.env'),
    'raw-Imaging-only':shutil.ignore_patterns('*.npy', 'suite2*', 
                                              'FaceCamera-*', 'RigCamera-*',
                                              '*.env'),
    'video-Imaging-only':shutil.ignore_patterns('*.ome.tif', 
                                                'FaceCamera-*', 'RigCamera-*',
                                                'db.npy', 'ops.npy', 
                                                'suite2*', '*.env'),
    'processed-Behavior':shutil.ignore_patterns('FaceCamera-imgs*', 'RigCamera-imgs*',
                                                '*.mp4', '*.avi','*.wmv',
                                                'TSerie*'),
    'processed-Behavior-wVids':shutil.ignore_patterns('FaceCamera-imgs*', 'RigCamera-imgs*',
                                                      'TSerie*'),
    'nwb':include_patterns('*.nwb'),
    'npy':include_patterns('*.npy'),
    'xml':include_patterns('*.xml'),
    'all':None,
    'to_build_nwb_bacci': shutil.ignore_patterns('FaceCamera-imgs*', 'RigCamera-imgs*',
                                                 '*.ome.tif', 'Reference*', 'CYCLE*','*.bin',
                                                 '*.mp4', '*.avi','*.wmv',
                                                 '*.env',  'reg_tif*', 
                                                 '*_output_*', "*.png"
                                                    ),
        }


if __name__=='__main__':

    import sys

    if sys.argv[-1] not in TYPES:
       print("""

       should be used as 
       python -m physion.utils.transfer.types SOURCE DEST type

       with type in :
       """)
       for key in TYPES:
           print(' - '+key)

    else:
        Source = sys.argv[-3]
        Dest = sys.argv[-2]
        Type = sys.argv[-1]

        shutil.copytree(Source, Dest, 
                        dirs_exist_ok=True,
                        ignore=TYPES[Type])
