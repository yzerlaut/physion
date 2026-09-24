"""
conversion of Bruker TSeries (tiffs) to h5 files (one per channel and plane)
"""
import os
import numpy as np
import h5py
import pytest

from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.files import get_files_with_extension
from physion.utils.compression.h5 import tiffs_to_h5, convert_to_h5
from physion.utils.compression.twoP import create_compressed_folder


@pytest.mark.parametrize('nplanes', [1, 3])
def test_bruker_parser(make_TSeries, nplanes):
    TS, movies = make_TSeries(nframes=10, nplanes=nplanes)
    xml = bruker_xml_parser(get_files_with_extension(TS, extension='.xml')[0])

    assert xml['channels'] == ['Ch1 Red', 'Ch2 Green']
    assert float(xml['settings']['framePeriod']) == 0.03
    for chan in xml['channels']:
        assert len(xml[chan]['tifFile']) == 10*nplanes
        assert sorted(np.unique(xml[chan]['depth_index'])) == list(range(nplanes))


def test_tiffs_to_h5_is_lossless(make_TSeries, tmp_path):
    TS, movies = make_TSeries(nframes=70, channels=('Ch2 Green',))
    xml = bruker_xml_parser(get_files_with_extension(TS, extension='.xml')[0])

    out = tiffs_to_h5(TS, xml['Ch2 Green']['tifFile'], str(tmp_path/'out.h5'),
                      batch_size=32, # -> partial last batch
                      with_ProgressBar=False)

    with h5py.File(out, 'r') as f:
        assert f['data'].dtype == np.uint16
        np.testing.assert_array_equal(f['data'][:], movies[('Ch2 Green', 0)])


@pytest.mark.parametrize('nplanes', [1, 2])
def test_convert_to_h5(make_TSeries, nplanes):
    TS, movies = make_TSeries(nframes=15, nplanes=nplanes)
    create_compressed_folder(TS, 'h5')
    convert_to_h5(TS)

    h5_folder = TS.replace('TSeries-', 'h5-')
    assert any(f.endswith('.xml') for f in os.listdir(h5_folder))
    for (chan, plane), movie in movies.items():
        fn = os.path.join(h5_folder, '%s-plane%i.h5' % (chan.replace(' ', '-'), plane))
        with h5py.File(fn, 'r') as f:
            np.testing.assert_array_equal(f['data'][:], movie)
