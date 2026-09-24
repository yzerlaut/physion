"""
naming conventions of the imaging folders: "TSeries-" (raw) and "h5-" (compressed)
"""
import os
import pytest

from physion.imaging.suite2p.preprocessing import find_imaging_folders,\
        is_TSeries_folder, is_h5_folder
from physion.utils.compression import twoP


def make_dirs(root, dirs):
    for d in dirs:
        os.makedirs(os.path.join(root, d))


def test_is_folder_helpers():
    assert is_TSeries_folder('/a/b/TSeries-001/')
    assert is_h5_folder('/a/TSeries-data/h5-001')
    assert not is_TSeries_folder('/a/TSeries-data/log8bit-001')
    assert not is_h5_folder('/a/b/data.h5')


def test_find_imaging_folders(tmp_path):
    make_dirs(tmp_path, ['day1/TSeries-A/suite2p/TSeries-inside',
                         'day1/h5-A',  # skipped: its TSeries is still there
                         'day1/h5-B',
                         'day1/log8bit-C', 'day1/lossless-D',
                         'day2/sub/h5-E'])
    rel = lambda folders: [os.path.relpath(f, tmp_path) for f in folders]

    assert rel(find_imaging_folders(tmp_path)) ==\
            ['day1/TSeries-A', 'day1/h5-B', 'day2/sub/h5-E']
    assert rel(find_imaging_folders(tmp_path/'day1', recursive=False)) ==\
            ['day1/TSeries-A', 'day1/h5-B']


def test_compression_finds_only_raw_TSeries(tmp_path):
    make_dirs(tmp_path, ['s1/TSeries-X/suite2p/TSeries-inside',
                         's1/log8bit-X', 's1/h5-X', 's1/nwb-X'])
    assert [os.path.relpath(f, tmp_path)\
                for f in twoP.find_TSeries_folders(tmp_path)] ==\
            ['s1/TSeries-X']


@pytest.mark.parametrize('key, folder_key', [('h5', 'h5'),
                                             ('8bit-LOG-mp4', 'log8bit'),
                                             ('16bit-avi (lossless)', 'lossless')])
def test_create_compressed_folder(tmp_path, key, folder_key):
    # a parent folder containing "TSeries" in its name should not be renamed
    TS = tmp_path/'my-TSeries-data'/'TSeries-X'
    make_dirs(TS, ['References'])
    (TS/'TSeries-X.xml').write_text('<xml/>')
    (TS/'frame_000001.ome.tif').write_text('')

    twoP.create_compressed_folder(str(TS), key)

    new = tmp_path/'my-TSeries-data'/('%s-X' % folder_key)
    assert sorted(os.listdir(new)) == ['References', 'TSeries-X.xml'] # no tiffs
    assert sorted(os.listdir(tmp_path)) == ['my-TSeries-data']


def test_create_compressed_folder_refuses_non_TSeries(tmp_path):
    make_dirs(tmp_path, ['log8bit-X'])
    with pytest.raises(ValueError):
        twoP.create_compressed_folder(str(tmp_path/'log8bit-X'), 'h5')


###########################################################################
####     physion.imaging.folders (the other modules use these)       #####
###########################################################################

from physion.imaging import folders


def test_compressed_folder_and_plane_file():
    assert folders.compressed_folder('/data/my-TSeries-exp/TSeries-001', 'h5') ==\
            '/data/my-TSeries-exp/h5-001'
    assert folders.compressed_folder('/data/TSeries-001/', 'log8bit') ==\
            '/data/log8bit-001'
    with pytest.raises(ValueError):
        folders.compressed_folder('/data/TSeries-exp/h5-001', 'nwb')
    assert folders.plane_file('/data/h5-001', 'Ch2 Green', 1, 'h5') ==\
            '/data/h5-001/Ch2-Green-plane1.h5'


def test_find_compressed_folders(tmp_path):
    make_dirs(tmp_path, ['s1/log8bit-X', 's1/h5-X', 's2/log8bit-Y/sub/log8bit-Z',
                         's2/TSeries-log8bit'])
    assert [os.path.relpath(f, tmp_path)\
                for f in folders.find_compressed_folders(tmp_path, 'log8bit')] ==\
            ['s1/log8bit-X', 's2/log8bit-Y']


@pytest.mark.parametrize('content, expected', [
    (['TSeries-001', 'h5-001', 'FaceCamera-imgs'], ['TSeries-001']),
    (['h5-001', 'FaceCamera-imgs'], ['h5-001']),     # after conversion
    (['TSeries-001', 'TSeries-002'], ['TSeries-001', 'TSeries-002']),
    (['FaceCamera-imgs'], []),
])
def test_session_imaging_folders(tmp_path, content, expected):
    make_dirs(tmp_path, content)
    (tmp_path/'TSeries-notes.txt').write_text('') # files are ignored
    assert [os.path.basename(f)\
                for f in folders.session_imaging_folders(tmp_path)] == expected
