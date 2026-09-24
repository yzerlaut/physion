"""
naming conventions of the 2P imaging folders

    - "TSeries-xxx"  : raw Bruker data (xml + tiffs), with its "suite2p/" output
    - "h5-xxx", "nwb-xxx", "binary-xxx", "log8bit-xxx", "lossless-xxx" :
            compressed versions of "TSeries-xxx" (see physion.utils.compression),
            created next to it, with its metadata files (xml, suite2p/, ...)

only the folder *names* follow the conventions,
    the parent folders can be named anything (even "my-TSeries-data/")
"""
import os


def is_TSeries_folder(folder):
    return os.path.basename(os.path.normpath(str(folder))).startswith('TSeries-')


def is_h5_folder(folder):
    return os.path.basename(os.path.normpath(str(folder))).startswith('h5-')


def compressed_folder(TS_folder, key):
    """
    "/any/path/TSeries-xxx" -> "/any/path/{key}-xxx"
    """
    folder = os.path.normpath(str(TS_folder))
    name = os.path.basename(folder)
    if 'TSeries' not in name:
        raise ValueError('"%s" is not a "TSeries-" folder' % TS_folder)
    return os.path.join(os.path.dirname(folder), name.replace('TSeries', key, 1))


def plane_file(folder, channel, plane, extension):
    """ file of a given channel and plane, e.g. ".../Ch2-Green-plane0.h5" """
    return os.path.join(str(folder), '%s-plane%i.%s' %\
                        (channel.replace(' ', '-'), plane, extension))


def find_TSeries_folders(folder):
    """
    raw "TSeries-" folders (no search inside them)
    """
    FOLDERS = []
    for root, subdirs, _ in os.walk(folder):
        FOLDERS += [os.path.join(root, d) for d in subdirs\
                            if is_TSeries_folder(d)]
        subdirs[:] = [d for d in subdirs if not is_TSeries_folder(d)]
    return sorted(FOLDERS)


def find_compressed_folders(folder, key='h5'):
    """
    "{key}-" folders (no search inside them)
    """
    FOLDERS = []
    for root, subdirs, _ in os.walk(folder):
        FOLDERS += [os.path.join(root, d) for d in subdirs\
                            if d.startswith(key+'-')]
        subdirs[:] = [d for d in subdirs if not d.startswith(key+'-')]
    return sorted(FOLDERS)


def find_imaging_folders(folder, recursive=True):
    """
    "TSeries-" and "h5-" folders (no search inside them)
        "h5-" folders are skipped if their "TSeries-" folder is still there
    """
    FOLDERS = []
    for root, subdirs, _ in os.walk(folder):
        for d in sorted(subdirs):
            if is_TSeries_folder(d) or\
                    (is_h5_folder(d) and\
                        (d.replace('h5-', 'TSeries-', 1) not in subdirs)):
                FOLDERS.append(os.path.join(root, d))
        subdirs[:] = [d for d in subdirs\
                        if not (is_TSeries_folder(d) or is_h5_folder(d))]
        if not recursive:
            break
    return sorted(FOLDERS)


def session_imaging_folders(datafolder):
    """
    imaging folder(s) of a recording session: the "TSeries" folder(s),
        otherwise the "h5-" folder(s) when the raw data were converted
    """
    subdirs = [d for d in sorted(os.listdir(datafolder))\
                    if os.path.isdir(os.path.join(datafolder, d))]
    TSeries = [d for d in subdirs if 'TSeries' in d]
    if len(TSeries)==0:
        TSeries = [d for d in subdirs if is_h5_folder(d)]
    return [os.path.join(datafolder, d) for d in TSeries]
