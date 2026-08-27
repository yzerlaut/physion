import os
from PIL import Image
import numpy as np

from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.paths import FOLDERS

from pynwb import NWBHDF5IO, NWBFile

from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO

from dateutil.tz import tzlocal
from datetime import datetime
from uuid import uuid4

def frame_generator(TS_folder, tiff_files, expected_shape=None, expected_dtype=None):
    """Yield one 2D (height, width) frame at a time, reading lazily from disk.
 
    Because this is a plain generator (not a generator of DataChunk objects),
    DataChunkIterator treats each yielded array as one chunk and stacks the
    chunks along `iter_axis` (0 here) to build the (n_frames, height, width)
    dataset incrementally -- see the "iterative write" tutorial.
    """
    for f in tiff_files:
        frame = np.array(Image.open(os.path.join(TS_folder, f)),
                   dtype='uint16')
 
        if frame.ndim != 2:
            raise ValueError(
                f"Expected a single-frame (2D) TIFF, got shape {frame.shape} "
                f"for file: {f}"
            )
        if expected_shape is not None and frame.shape != expected_shape:
            raise ValueError(
                f"Frame shape {frame.shape} in {f} does not match the "
                f"expected shape {expected_shape} (from the first file)."
            )
        if expected_dtype is not None and frame.dtype != expected_dtype:
            raise ValueError(
                f"Frame dtype {frame.dtype} in {f} does not match the "
                f"expected dtype {expected_dtype} (from the first file)."
            )
 
        yield frame
 
 
def peek_frame_shape_and_dtype(TS_folder, tiff_files):
    """Read only the first file to learn frame shape/dtype ahead of time.
 
    Knowing these lets us pass an exact `maxshape`/`dtype` to
    DataChunkIterator instead of having it guess, and avoids an
    open-ended (resizable-forever) dataset.
    """
    first = np.array(Image.open(os.path.join(TS_folder, tiff_files[0])),
                   dtype='uint16')
    if first.ndim != 2:
        raise ValueError(
            f"Expected a single-frame (2D) TIFF, got shape {first.shape} "
            f"for file: {tiff_files[0]}"
        )
    return first.shape, first.dtype


###########################################################################
  
def build_nwbfile(
    TS_folder,
    tiff_files,
    session_description,
    identifier,
    session_start_time,
    imaging_rate_hz,
    device_name="Microscope",
    device_description="Two-photon microscope",
    device_manufacturer=None,
    excitation_lambda=920.0,
    emission_lambda=510.0,
    indicator="GCaMP6f",
    location="V1",
    buffer_size=10,
    compression="gzip",
):
    """Build an NWBFile with a TwoPhotonSeries backed by a DataChunkIterator
    that reads `tiff_files` lazily, one frame at a time.
    """
    if len(tiff_files) == 0:
        raise ValueError("tiff_files is empty.")
 
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
    )
 
    device = nwbfile.create_device(
        name=device_name,
        description=device_description,
        manufacturer=device_manufacturer,
    )
 
    optical_channel = OpticalChannel(
        name="OpticalChannel",
        description="2-photon optical channel",
        emission_lambda=emission_lambda,
    )
 
    imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane",
        optical_channel=optical_channel,
        imaging_rate=imaging_rate_hz,
        description="Imaging plane reconstructed from a series of single-frame TIFFs",
        device=device,
        excitation_lambda=excitation_lambda,
        indicator=indicator,
        location=location,
    )
 
    # Peek at the first frame only, to pin down shape/dtype for maxshape.
    frame_shape, frame_dtype = peek_frame_shape_and_dtype(TS_folder, tiff_files)
    n_frames = len(tiff_files)
 
    data_iterator = DataChunkIterator(
        data=frame_generator(TS_folder, tiff_files, 
                             expected_shape=frame_shape, 
                             expected_dtype=frame_dtype),
        iter_axis=0,
        maxshape=(n_frames,) + frame_shape,
        dtype=np.dtype('uint16'),
        buffer_size=buffer_size,  # frames buffered per HDF5 write -- raise for speed/fewer, larger writes
    )
 
    # Wrap in H5DataIO to get chunked + compressed storage on disk (optional
    # but recommended for large movies). H5DataIO is chunk-iterator-aware.
    wrapped_data = H5DataIO(data=data_iterator, compression=compression)
 
    two_photon_series = TwoPhotonSeries(
        name="TwoPhotonSeries",
        description="Raw 2-photon imaging series assembled from single-frame TIFF files",
        data=wrapped_data,
        imaging_plane=imaging_plane,
        rate=imaging_rate_hz,   # use `timestamps=...` instead of `rate` if frames are not evenly spaced
        starting_time=0.0,
        unit="a.u.",
    )
 
    nwbfile.add_acquisition(two_photon_series)
 
    return nwbfile
 
def write_nwb(nwbfile, out_path):
    with NWBHDF5IO(out_path, mode="w") as io:
        io.write(nwbfile)

def convert_to_nwb(TS_folder):

    xml_file = get_files_with_extension(TS_folder, 
                                        extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    if not os.path.isdir(os.path.join(TS_folder.replace('TSeries', 'nwb'))):
        os.mkdir(os.path.join(TS_folder.replace('TSeries', 'nwb')))

    print('\n Analyzing: "%s" ' % TS_folder)
    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        movie_rate = 1./float(xml['settings']['framePeriod'])
        FILES = xml[chan]['tifFile']

        DICT = {'compression':'None'}
        
        for p in np.unique(xml[chan]['depth_index']):

            plane_cond = (xml[chan]['depth_index']==p)

            vid_name = os.path.join(TS_folder.replace('TSeries', 'nwb'),
                                     '%s-plane%i.nwb' %\
                                    (chan.replace(' ','-'), p))

            nwbfile = build_nwbfile(
                TS_folder,
                tiff_files=FILES[plane_cond],
                session_description="2-photon imaging session",
                identifier=str(uuid4()),
                session_start_time=datetime.now(tzlocal()),
                imaging_rate_hz=movie_rate,       # set to your actual frame rate
                device_manufacturer="Bruker",   # e.g. "Bruker", "Thorlabs", ...
                excitation_lambda=920.0,
                emission_lambda=510.0,
                indicator="GCaMP6s",
                location="V1",
                buffer_size=10,
            )
        
            write_nwb(nwbfile, vid_name)
            print(f" [ok] succesfully wrote {len(FILES[plane_cond])} frames to ", vid_name)

        np.save(os.path.join(TS_folder.replace('TSeries', 'nwb'), 
                             '%s-summary.npy'%chan.replace(' ','-')),
                DICT)
        print(' [ok] Frames-summary.npy succesfully created !')
