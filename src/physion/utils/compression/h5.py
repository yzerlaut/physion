import os
import numpy as np
from PIL import Image
import h5py

from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO

from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.paths import FOLDERS

from physion.utils.compression.nwb import peek_frame_shape_and_dtype


def tiffs_to_h5(
    TS_folder,
    tiff_files,
    out_path,
    dataset_key="data",
    compression="gzip",
    batch_size=32,
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
            return write_start + len(buffer)
 
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


def convert_to_h5(TS_folder):

    xml_file = get_files_with_extension(TS_folder, 
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    if not os.path.isdir(os.path.join(TS_folder.replace('TSeries', 'h5'))):
        os.mkdir(os.path.join(TS_folder.replace('TSeries', 'h5')))

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        movie_rate = 1./float(xml['settings']['framePeriod'])
        FILES = xml[chan]['tifFile']

        for p in np.unique(xml[chan]['depth_index']):

            plane_cond = (xml[chan]['depth_index']==p)

            vid_name = os.path.join(TS_folder.replace('TSeries', 'h5'),
                                     '%s-plane%i.h5' %\
                                    (chan.replace(' ','-'), p))

            h5_file = tiffs_to_h5(
                TS_folder,
                FILES[plane_cond],
                vid_name)
        
            print(f" [ok] succesfully wrote {len(FILES[plane_cond])} frames to ", vid_name)

        # np.save(os.path.join(TS_folder.replace('TSeries', 'h5'), 
        #                      '%s-summary.npy'%chan.replace(' ','-')),
        #         DICT)
        # print(' [ok] Frames-summary.npy succesfully created !')
