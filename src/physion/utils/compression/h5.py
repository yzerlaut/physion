import os, shutil, stat
import numpy as np
from PIL import Image
import h5py

from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO

from physion.imaging.folders import compressed_folder, plane_file,\
        find_TSeries_folders, is_TSeries_folder
from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.paths import FOLDERS
from physion.utils.progressBar import printProgressBar

from physion.utils.compression.nwb import peek_frame_shape_and_dtype


def tiffs_to_h5(
    TS_folder,
    tiff_files,
    out_path,
    dataset_key="data",
    compression="gzip",
    batch_size=32,
    with_ProgressBar=True,
):
    """Write `tiff_files` into out_path as an HDF5 dataset under `dataset_key`,
    shape (n_frames, height, width), reading/writing `batch_size` frames at a
    time so memory use stays bounded regardless of how many files there are.
    """
    if len(tiff_files) == 0:
        raise ValueError("tiff_files is empty.")
 
    frame_shape, dtype = peek_frame_shape_and_dtype(TS_folder, 
                                                    tiff_files)
    n_frames = len(tiff_files)
    chunk_frames = min(batch_size, n_frames)
 
    with h5py.File(out_path, "w") as h5f:
        dset = h5f.create_dataset(
            dataset_key,
            shape=(n_frames,) + frame_shape,
            dtype=dtype,
            chunks=(chunk_frames,) + frame_shape,
            compression=compression,
        )
 
        buffer = []
        write_start = 0
 
        def flush(buffer, write_start):
            if not buffer:
                return write_start
            stacked = np.stack(buffer)
            dset[write_start:write_start + len(buffer)] = stacked
            if with_ProgressBar:
                printProgressBar(write_start + len(buffer), n_frames,
                                 prefix='    writing h5:',
                                 suffix='(%i/%i frames)' %\
                                    (write_start + len(buffer), n_frames))
            return write_start + len(buffer)

        if with_ProgressBar:
            printProgressBar(0, n_frames, prefix='    writing h5:',
                             suffix='(0/%i frames)' % n_frames)

        for i, f in enumerate(tiff_files):

            frame = np.array(Image.open(os.path.join(TS_folder, f)),
                    dtype='uint16')
 
            if frame.ndim != 2:
                raise ValueError(
                    f"Expected a single-frame (2D) TIFF, got shape {frame.shape} "
                    f"for file: {f}"
                )
            if frame.shape != frame_shape:
                raise ValueError(
                    f"Frame shape {frame.shape} in {f} does not match the "
                    f"expected shape {frame_shape} (from the first file)."
                )
            if frame.dtype != dtype:
                raise ValueError(
                    f"Frame dtype {frame.dtype} in {f} does not match the "
                    f"expected dtype {dtype} (from the first file)."
                )
 
            buffer.append(frame)
 
            if len(buffer) == batch_size:
                write_start = flush(buffer, write_start)
                buffer = []
 
        write_start = flush(buffer, write_start)
        assert write_start == n_frames
 
    return out_path


def get_xml_file(TS_folder):
    xml_files = get_files_with_extension(TS_folder, extension='.xml')
    if len(xml_files)==0:
        raise FileNotFoundError('no xml file in "%s"' % TS_folder)
    return xml_files[0]


class MissingTiffsError(Exception):
    """ some tiffs of the Bruker xml are not in the "TSeries-" folder """
    pass


def check_tiffs_before_h5(TS_folder):
    """ raises MissingTiffsError if tiffs of the xml are missing """
    missing, _ = check_tiff_coverage(TS_folder,
                                     build_conversion_plan(TS_folder))
    if len(missing)>0:
        raise MissingTiffsError(
            '%i tiffs of the xml are missing in "%s" (e.g. %s)' %\
                (len(missing), TS_folder, sorted(missing)[0]))


def convert_to_h5(TS_folder):

    # nothing is written if tiffs are missing
    check_tiffs_before_h5(TS_folder)

    xml_file = get_xml_file(TS_folder)
    xml = bruker_xml_parser(xml_file)

    if not os.path.isdir(compressed_folder(TS_folder, 'h5')):
        os.mkdir(compressed_folder(TS_folder, 'h5'))

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        movie_rate = 1./float(xml['settings']['framePeriod'])
        FILES = xml[chan]['tifFile']

        for p in np.unique(xml[chan]['depth_index']):

            plane_cond = (xml[chan]['depth_index']==p)

            vid_name = plane_file(compressed_folder(TS_folder, 'h5'), chan, p, 'h5')

            h5_file = tiffs_to_h5(
                TS_folder,
                FILES[plane_cond],
                vid_name)
        
            print(f" [ok] succesfully wrote {len(FILES[plane_cond])} frames to ", vid_name)

        # np.save(os.path.join(compressed_folder(TS_folder, 'h5'), 
        #                      '%s-summary.npy'%chan.replace(' ','-')),
        #         DICT)
        # print(' [ok] Frames-summary.npy succesfully created !')


###########################################################################
####   checks before removing the raw data (the "TSeries-" folder)    #####
###########################################################################

H5_KEY = 'data'


def is_tiff(filename):
    return filename.lower().endswith(('.tif', '.tiff'))


def build_conversion_plan(TS_folder, h5_folder=None):
    """
    returns a list of (h5_file, tiff_files), one per channel and plane
        (same naming than convert_to_h5)
    """
    if h5_folder is None:
        h5_folder = compressed_folder(TS_folder, 'h5')
    xml_file = get_xml_file(TS_folder)
    xml = bruker_xml_parser(xml_file)

    plan = []
    for chan in xml['channels']:
        FILES = np.array(xml[chan]['tifFile'])
        depth_index = np.array(xml[chan]['depth_index'])
        for p in np.unique(depth_index):
            plan.append((plane_file(h5_folder, chan, p, 'h5'),
                         list(FILES[depth_index==p])))
    return plan


def check_tiff_coverage(TS_folder, plan):
    """
    returns the sets of:
        - "missing" tiffs: in the xml but not in the folder
        - "unplanned" tiffs: in the folder but not in the xml
    """
    planned = set(f for _, tiffs in plan for f in tiffs)
    present = set(f for f in os.listdir(TS_folder) if is_tiff(f))
    return planned-present, present-planned


def verify_h5(TS_folder, tiff_files, h5_file, batch_size=32):
    """ pixel-exact comparison of every h5 frame with its tiff
        returns None if identical, otherwise the problem (str) """
    try:
        with h5py.File(h5_file, 'r') as f:
            if H5_KEY not in f:
                return 'no "%s" key' % H5_KEY
            dset = f[H5_KEY]
            if dset.shape[0]!=len(tiff_files):
                return '%i frames in h5 vs %i tiffs' % (dset.shape[0],
                                                       len(tiff_files))
            n = len(tiff_files)
            for i0 in range(0, n, batch_size):
                frames = dset[i0:i0+batch_size]
                for frame, tiff in zip(frames, tiff_files[i0:i0+batch_size]):
                    ref = np.array(Image.open(os.path.join(TS_folder, tiff)))
                    if (ref.shape!=frame.shape) or\
                            (not np.array_equal(ref, frame)):
                        print()
                        return 'frame mismatch with "%s"' % tiff
                i1 = min([n, i0+batch_size])
                printProgressBar(i1, n, prefix='    checking h5:',
                                 suffix='(%i/%i frames)' % (i1, n))
    except Exception as e:
        print()
        return 'unreadable h5 (%s)' % e
    return None # no error


def copy_non_tiff_content(TS_folder, h5_folder):
    """ copy xml, env, References/, suite2p/, ...
        and check the copies on file sizes """
    errors = []
    for item in os.listdir(TS_folder):
        src, dst = os.path.join(TS_folder, item), os.path.join(h5_folder, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            pairs = [(os.path.join(root, f),
                      os.path.join(dst, os.path.relpath(root, src), f))\
                        for root, _, files in os.walk(src) for f in files]
        elif not is_tiff(item):
            shutil.copy2(src, dst)
            pairs = [(src, dst)]
        else:
            continue
        for s, d in pairs:
            if (not os.path.isfile(d)) or\
                    (os.path.getsize(s)!=os.path.getsize(d)):
                errors.append('copy failed for "%s"' % s)
    return errors


def remove_readonly(func, path, _):
    """ on network shares, some files can be read-only """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_failed_h5_folder(TS_folder):
    """ after a conversion that failed partway (the raw data is still there) """
    h5_folder = compressed_folder(TS_folder, 'h5')
    if os.path.isdir(h5_folder):
        shutil.rmtree(h5_folder, onerror=remove_readonly)
        print('     [!!] partial "%s" removed' % h5_folder)


def remove_TSeries_if_converted(TS_folder):
    """
    removes the "TSeries-" folder after its conversion to h5, only if:
        - every tiff of the folder is referenced in the Bruker xml file
        - every h5 file opens and matches its tiffs frame by frame (pixel-exact)
        - every non-tiff file/folder (xml, env, References/, suite2p/, ...)
            is copied to the "h5-" folder (checked on file sizes)

    returns (removed, problems)
    """
    h5_folder = compressed_folder(TS_folder, 'h5')
    plan = build_conversion_plan(TS_folder, h5_folder)

    problems = []
    missing, unplanned = check_tiff_coverage(TS_folder, plan)
    if len(missing)>0:
        problems.append('%i tiffs of the xml are missing (e.g. %s)' %\
                            (len(missing), sorted(missing)[0]))
    if len(unplanned)>0:
        problems.append('%i tiffs are not referenced in the xml (e.g. %s)' %\
                            (len(unplanned), sorted(unplanned)[0]))
    if len(problems)==0:
        for h5_file, tiffs in plan:
            error = verify_h5(TS_folder, tiffs, h5_file)
            if error is None:
                print('     [ok] %s verified' % os.path.basename(h5_file))
            else:
                problems.append('%s: %s' % (os.path.basename(h5_file), error))
    if len(problems)==0:
        problems += copy_non_tiff_content(TS_folder, h5_folder)

    if len(problems)>0:
        for p in problems:
            print('     [!!] %s' % p)
        print('     [!!] "%s" NOT removed' % TS_folder)
        return False, problems

    shutil.rmtree(TS_folder, onerror=remove_readonly)
    print('     [ok] "%s" removed' % TS_folder)
    return True, []


###########################################################################
####                full pipeline (used by the GUI and CLI)            #####
###########################################################################

def create_h5_folder(TS_folder):
    """ the "h5-" folder with the non-tiff content of the "TSeries-" folder
        (same as create_compressed_folder in twoP.py, not imported to avoid
         a circular import and the Qt dependency on the command line) """
    shutil.copytree(TS_folder, compressed_folder(TS_folder, 'h5'),
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('*.ome.tif', 'CYCLE*', '*.bin'))


def convert_TSeries_folder(TS_folder, remove_raw=False):
    """
    checks the tiffs, converts to h5 and (optionally) removes the raw data

    returns (converted, removed, message)
        message: why it failed (or why the raw data was not removed)
    """
    try:
        check_tiffs_before_h5(TS_folder)
    except MissingTiffsError as e:
        print(' [!!] %s --> skipped' % e)
        return False, False, str(e)
    except Exception as e:
        print(' [!!] "%s" not converted, %s: %s --> skipped' %\
                (TS_folder, type(e).__name__, e))
        return False, False, '%s: %s' % (type(e).__name__, e)

    # only a folder created here is removed if the conversion fails
    new_h5_folder = not os.path.isdir(compressed_folder(TS_folder, 'h5'))
    try:
        create_h5_folder(TS_folder)
        convert_to_h5(TS_folder)
    except Exception as e:
        print('\n [!!] conversion of "%s" failed, %s: %s --> skipped' %\
                (TS_folder, type(e).__name__, e))
        if new_h5_folder:
            remove_failed_h5_folder(TS_folder)
        return False, False, 'conversion failed, %s: %s' % (type(e).__name__, e)

    removed, message = False, ''
    if remove_raw:
        try:
            # only after checking the conversion (see above)
            removed, problems = remove_TSeries_if_converted(TS_folder)
            message = '; '.join(problems)
        except Exception as e:
            # the h5 folder is kept: the raw data might be partially removed
            print('\n [!!] removal of "%s" failed, %s: %s' %\
                    (TS_folder, type(e).__name__, e))
            message = 'removal failed, %s: %s' % (type(e).__name__, e)
    return True, removed, message


if __name__=='__main__':

    import argparse, datetime

    parser=argparse.ArgumentParser(description="""
    converts the "TSeries-" folders (Bruker tiffs) found in "folder"
        to "h5-" folders (one lossless h5 file per channel and plane)

    writes two logs (in "folder", or its parent for a "TSeries-" folder):
        h5-compression_<date>_succeeded.txt
        h5-compression_<date>_failed.txt
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("folder",
                        help='a "TSeries-" folder or a folder containing some')
    parser.add_argument("--remove_raw", "--rm", "--delete",
                        dest="remove_raw",
                        help='remove each "TSeries-" folder after checking that\n'
                             ' - all its tiffs are in the xml file\n'
                             ' - the h5 files match the tiffs, frame by frame (pixel-exact)\n'
                             ' - all its other files are copied to the "h5-" folder',
                        action="store_true")
    parser.add_argument("--log_dir", default=None,
                        help='folder of the log files (default: see above)')
    args = parser.parse_args()

    if is_TSeries_folder(args.folder):
        folders = [os.path.normpath(args.folder)]
        log_dir = os.path.dirname(folders[0])
    else:
        folders = find_TSeries_folders(args.folder)
        log_dir = args.folder
    if args.log_dir is not None:
        log_dir = args.log_dir

    if len(folders)==0:
        print('\n no "TSeries-" folder found in "%s" \n' % args.folder)

    else:
        os.makedirs(log_dir, exist_ok=True)
        date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logs = {key:os.path.join(log_dir, 'h5-compression_%s_%s.txt' % (date, key))\
                    for key in ['succeeded', 'failed']}
        # created now and filled during the loop (kept if interrupted)
        for key in logs:
            with open(logs[key], 'w') as f:
                f.write('# h5 compression %s -- %s (remove_raw=%s)\n' %\
                        (key, date, args.remove_raw))

        summary = []
        for folder in folders:
            print('\n - processing', folder, ' [...]')
            converted, removed, message = convert_TSeries_folder(folder,
                                                    remove_raw=args.remove_raw)
            if converted:
                line = folder
                if args.remove_raw:
                    line += '\t raw removed' if removed else\
                            '\t raw NOT removed: %s' % message
            else:
                line = '%s\t %s' % (folder, message)
            with open(logs['succeeded' if converted else 'failed'], 'a') as f:
                f.write(line+'\n')
            summary.append((folder, converted, removed))

        print('\n ---- summary ----')
        for folder, converted, removed in summary:
            status = 'converted' if converted else 'NOT converted'
            if args.remove_raw and converted:
                status += ', raw %s' % ('removed' if removed else 'NOT removed')
            print('  %s : %s' % (folder, status))
        print('\n logs: \n  %s\n  %s\n' % (logs['succeeded'], logs['failed']))
