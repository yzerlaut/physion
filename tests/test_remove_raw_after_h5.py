"""
removal of the raw data ("TSeries-" folder) after the conversion to h5,
    only when the conversion is checked (see physion.utils.compression.h5)
"""
import os
import numpy as np
import h5py
import pytest
from PIL import Image

from physion.utils.compression.h5 import convert_to_h5, remove_TSeries_if_converted
from physion.utils.compression.twoP import create_compressed_folder


def converted_TSeries(make_TSeries):
    TS, movies = make_TSeries(nframes=12, nplanes=2)
    # suite2p output, with the binary file that create_compressed_folder skips
    os.makedirs(os.path.join(TS, 'suite2p', 'plane0'))
    np.zeros(100, dtype=np.int16).tofile(os.path.join(TS, 'suite2p', 'plane0', 'data.bin'))
    create_compressed_folder(TS, 'h5')
    convert_to_h5(TS)
    return TS, TS.replace('TSeries-', 'h5-')


def test_removed_when_checked(make_TSeries):
    TS, h5_folder = converted_TSeries(make_TSeries)
    removed, problems = remove_TSeries_if_converted(TS)
    assert removed and problems == []
    assert not os.path.exists(TS)
    assert any(f.endswith('.xml') for f in os.listdir(h5_folder))
    assert os.path.getsize(os.path.join(h5_folder, 'suite2p', 'plane0', 'data.bin')) == 200


def corrupt_one_pixel(TS, h5_folder):
    fn = os.path.join(h5_folder, 'Ch2-Green-plane1.h5')
    with h5py.File(fn, 'r+') as f:
        f['data'][5, 3, 3] += 1

def add_stray_tiff(TS, h5_folder):
    Image.fromarray(np.zeros((12, 16), np.uint16)).save(os.path.join(TS, 'stray.ome.tif'))

def delete_one_h5(TS, h5_folder):
    os.remove(os.path.join(h5_folder, 'Ch1-Red-plane0.h5'))

@pytest.mark.parametrize('problem', [corrupt_one_pixel, add_stray_tiff, delete_one_h5])
def test_kept_when_not_checked(make_TSeries, problem):
    TS, h5_folder = converted_TSeries(make_TSeries)
    problem(TS, h5_folder)
    removed, problems = remove_TSeries_if_converted(TS)
    assert not removed and len(problems)>0
    assert os.path.isdir(TS) and len([f for f in os.listdir(TS) if f.endswith('.tif')])>=48


@pytest.mark.parametrize('fmt, removed', [('h5', True), ('16bit-avi (lossless)', False)])
def test_conversion_window_rm_raw(gui, make_TSeries, monkeypatch, fmt, removed):
    import physion.utils.compression.twoP as twoP
    TS, _ = make_TSeries(nframes=5)
    monkeypatch.setattr(twoP, 'convert_to_16bit_avi', lambda f: None) # (not tested here)
    window = gui.imaging_to_movie_gui()
    window.source_folder = os.path.dirname(TS)
    window.typeBox.setCurrentText(fmt)
    window.rm.setChecked(True)
    window.run_imaging_to_movie()
    assert os.path.exists(TS) != removed
    assert gui.slot_errors == []
