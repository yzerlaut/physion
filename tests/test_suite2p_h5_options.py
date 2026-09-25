"""
suite2p options for "h5-" folders:
    a virtual dataset interleaves the per-channel/per-plane h5 files
    in the frame order expected by suite2p (t: plane0-chan0, plane0-chan1, ...)
"""
import os
import numpy as np
import h5py
import pytest

from physion.imaging.suite2p.preprocessing import build_suite2p_options, H5_INPUT
from physion.utils.compression.h5 import convert_to_h5
from physion.utils.compression.twoP import create_compressed_folder


def settings(v1=True, subsampling=None):
    s = {'v1':v1, 'cell_diameter':20., 'subsampling':subsampling is not None}
    if subsampling is not None:
        s['subsampling_iStart'], s['subsampling_iStop'],\
                s['subsampling_step'] = subsampling
    return s


def make_h5_folder(make_TSeries, **kwargs):
    TS, movies = make_TSeries(**kwargs)
    create_compressed_folder(TS, 'h5')
    convert_to_h5(TS)
    return TS.replace('TSeries-', 'h5-'), movies


@pytest.mark.parametrize('nplanes, channels, subsampling', [
    (1, ('Ch1 Red', 'Ch2 Green'), None),
    (2, ('Ch1 Red', 'Ch2 Green'), None),
    (1, ('Ch2 Green',), None),
    (2, ('Ch1 Red', 'Ch2 Green'), (3, 17, 2)),
])
def test_h5_db_and_interleaving(make_TSeries, nplanes, channels, subsampling):
    folder, movies = make_h5_folder(make_TSeries, nframes=20,
                                    nplanes=nplanes, channels=channels)
    build_suite2p_options(folder, settings(subsampling=subsampling))

    db = np.load(os.path.join(folder, 'db.npy'), allow_pickle=True).item()
    assert db['input_format'] == 'h5'
    assert db['file_list'] == [H5_INPUT]
    assert db['nplanes'] == nplanes
    assert db['nchannels'] == len(channels)
    # "Ch2 Green" is the functional channel
    assert channels[db['functional_chan']-1] == 'Ch2 Green'

    frames = range(20)[slice(*subsampling)] if subsampling else range(20)
    with h5py.File(os.path.join(folder, H5_INPUT), 'r') as f:
        data = f['data'][:]
    ncp = nplanes*len(channels)
    assert data.shape[0] == len(frames)*ncp
    for it, t in enumerate(frames):
        for p in range(nplanes):
            for ic, c in enumerate(channels):
                np.testing.assert_array_equal(data[it*ncp+p*len(channels)+ic],
                                              movies[(c, p)][t])


def test_TSeries_db_unchanged(make_TSeries):
    TS, _ = make_TSeries(nframes=5)
    build_suite2p_options(TS, settings())
    db = np.load(os.path.join(TS, 'db.npy'), allow_pickle=True).item()
    assert db == {'data_path':[TS]}
    assert not os.path.isfile(os.path.join(TS, H5_INPUT))


def test_legacy_suite2p_ops(make_TSeries):
    """ suite2p<1.0: the input is read from "data_path", a folder with
        only the virtual dataset (suite2p<1.0 reads all the h5 files of the
        folder; its "h5py" key changed format across the 0.x versions) """
    folder, _ = make_h5_folder(make_TSeries, nframes=5)
    build_suite2p_options(folder, settings(v1=False))
    for f in ['ops.npy', 'db.npy']: # (db overrides ops in suite2p)
        ops = np.load(os.path.join(folder, f), allow_pickle=True).item()
        assert ops['input_format'] == 'h5'
        assert ops['h5py'] == []
        assert len(ops['data_path']) == 1
        assert [f for f in os.listdir(ops['data_path'][0]) if f.endswith('.h5')] ==\
                [os.path.basename(H5_INPUT)]
        assert ops['save_path0'] == folder
    assert ops['functional_chan'] == 2
