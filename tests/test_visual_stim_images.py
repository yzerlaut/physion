"""
images and virtual saccadic explorations (VSE) of the protocols stored in visual-stim.npy
    ("protocol_id-%i-Image-ID-%i" and "protocol_id-%i-VSE-%i" keys) and in the NWB files
"""
import numpy as np

from physion.visual_stim.stimuli.images import NI_BANK
from physion.visual_stim.build import build_stim, add_images_to_experiment, \
    image_key, is_image_key, add_vse_to_experiment, vse_key, is_vse_key

BASE = {"Presentation": "Stimuli-Sequence", "Screen": "Dell-2020-low-resolution",
        "no-window": True, "presentation-duration": 1, "presentation-prestim-period": 1,
        "presentation-poststim-period": 1, "presentation-interstim-period": 1, "N-repeat": 2}

IMAGES = {"NI_FOLDER": NI_BANK, "Image-ID-1": "1.jpg", "Image-ID-2": "2.jpg", "Image-ID-3": "3.jpg"}


def test_image_keys():
    assert image_key(3, 0) == 'protocol_id-3-Image-ID-0'
    assert is_image_key(image_key(3, 0))
    assert not is_image_key('protocol_id') and not is_image_key('Image-ID')


def test_natural_images_in_experiment():
    stim = build_stim(dict(BASE, Stimulus='images', **IMAGES, contrast=0.8))
    add_images_to_experiment(stim)
    keys = sorted(k for k in stim.experiment if is_image_key(k))
    assert keys == [image_key(0, i) for i in range(1, 4)]
    img = stim.experiment[image_key(0, 1)]
    assert img.dtype == np.uint8
    assert img.shape == tuple(stim.screen['resolution'][::-1])    # height x width, as the movie
    # the movie frame is the stored image with the contrast and background of the episode
    ep = list(stim.experiment['Image-ID']).index(1)
    frame = np.rot90(stim.get_image(ep), k=1)
    expected = stim.experiment['bg-color'][ep] + stim.experiment['contrast'][ep] * (img / 255. - 0.5)
    inside = np.rot90(((stim.x - stim.experiment['x-center'][ep])**2 +
                       (stim.z - stim.experiment['y-center'][ep])**2) < stim.experiment['radius'][ep]**2, k=1)
    np.testing.assert_allclose(frame[inside], expected[inside], atol=1. / 255)


def test_multiprotocol_and_protocols_without_images():
    multi = {"Presentation": "multiprotocol", "Screen": "Dell-2020-low-resolution", "no-window": True,
             "presentation-prestim-period": 1, "presentation-poststim-period": 1}
    for i, sub in [(1, dict(BASE, Stimulus='grating')),
                   (2, dict(BASE, Stimulus='images', NI_FOLDER=NI_BANK, **{"Image-ID-1": "1.jpg", "Image-ID-2": "2.jpg"}))]:
        multi['Protocol-%i' % i] = sub['Stimulus']
        for k, v in sub.items():
            multi['Protocol-%i-%s' % (i, k)] = v
    stim = build_stim(multi)
    add_images_to_experiment(stim)
    assert sorted(k for k in stim.experiment if is_image_key(k)) == [image_key(1, 1), image_key(1, 2)]

    grating = build_stim(dict(BASE, Stimulus='grating'))
    add_images_to_experiment(grating)
    assert not any(is_image_key(k) for k in grating.experiment)


def test_images_in_NWB(tmp_path):
    """ visual-stim.npy with images -> NWB file (stimulus templates) -> read back """
    import datetime, types
    import pynwb
    from dateutil.tz import tzlocal
    from physion.assembling.nwb import add_visual_stimulation
    from physion.analysis.modalities.visual_stim import VisualStimMixin

    # a recording folder with the visual-stim.npy of an images protocol
    stim = build_stim(dict(BASE, Stimulus='images', **IMAGES))
    add_images_to_experiment(stim)
    np.save(tmp_path / 'visual-stim.npy', stim.experiment)

    args = types.SimpleNamespace(datafolder=str(tmp_path), modalities=['VisualStim'],
                                 force_to_visualStimTimestamps=True, exclude_episodes=[],
                                 indices_forced=None, times_forced=None, durations_forced=None,
                                 max_episode=-1, photodiode_sampling=1000.,
                                 reverse_photodiodeSignal=False, verbose=False)
    nwbfile = pynwb.NWBFile(session_description='test', identifier='test',
                            session_start_time=datetime.datetime.now(tzlocal()))
    add_visual_stimulation(nwbfile, {'VisualStim': True}, None, args)
    with pynwb.NWBHDF5IO(str(tmp_path / 'test.nwb'), 'w') as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(str(tmp_path / 'test.nwb'), 'r') as io:
        data = types.SimpleNamespace(nwbfile=io.read())
        # the episode parameters are still there, without the images
        assert 'Image-ID' in data.nwbfile.stimulus
        assert not any(is_image_key(k) for k in data.nwbfile.stimulus)
        for image_id in range(1, 4):
            img = VisualStimMixin.get_visual_stim_image(data, 0, image_id)
            np.testing.assert_array_equal(img, stim.experiment[image_key(0, image_id)])
        assert VisualStimMixin.get_visual_stim_image(data, 0, 0) is None
        assert VisualStimMixin.get_visual_stim_image(data, 0, 5) is None
        assert VisualStimMixin.get_visual_stim_image(data, 1, 0) is None


VSE = {"seed": 3, "min-saccade-duration": 0.1, "max-saccade-duration": 1.0, "saccade-amplitude": 200.}


def test_vse_keys():
    assert vse_key(3, 0) == 'protocol_id-3-VSE-0'
    assert is_vse_key(vse_key(3, 0)) and not is_image_key(vse_key(3, 0))
    assert not is_vse_key(image_key(3, 0))


def test_vse_in_experiment():
    """ images-VSE and grating-VSE in a multiprotocol, with a grating (no VSE) """
    multi = {"Presentation": "multiprotocol", "Screen": "Dell-2020-low-resolution", "no-window": True,
             "presentation-prestim-period": 1, "presentation-poststim-period": 1}
    subs = [dict(BASE, Stimulus='grating'),
            dict(BASE, Stimulus='images-VSE', **IMAGES, **VSE),
            dict(BASE, Stimulus='grating-VSE', **dict(VSE, seed=5))]
    for i, sub in enumerate(subs, start=1):
        multi['Protocol-%i' % i] = sub['Stimulus']
        for k, v in sub.items():
            multi['Protocol-%i-%s' % (i, k)] = v
    stim = build_stim(multi)
    add_vse_to_experiment(stim)
    assert sorted(k for k in stim.experiment if is_vse_key(k)) == [vse_key(1, 3), vse_key(2, 5)]
    for (protocol_id, seed) in [(1, 3), (2, 5)]:
        stored = stim.experiment[vse_key(protocol_id, seed)]
        for key in ['t', 'x', 'y', 'max_amplitude']:
            np.testing.assert_array_equal(stored[key], stim.STIM[protocol_id].vse[key])


def test_vse_in_NWB(tmp_path):
    """ visual-stim.npy with a VSE -> NWB file (stimulus template) -> read back """
    import datetime, types
    import pynwb
    from dateutil.tz import tzlocal
    from physion.assembling.nwb import add_visual_stimulation
    from physion.analysis.modalities.visual_stim import VisualStimMixin

    stim = build_stim(dict(BASE, Stimulus='images-VSE', NI_FOLDER=NI_BANK,
                           **dict(VSE, **{"Image-ID-1": "1.jpg", "Image-ID-2": "2.jpg"})))
    add_images_to_experiment(stim)
    add_vse_to_experiment(stim)
    np.save(tmp_path / 'visual-stim.npy', stim.experiment)

    args = types.SimpleNamespace(datafolder=str(tmp_path), modalities=['VisualStim'],
                                 force_to_visualStimTimestamps=True, exclude_episodes=[],
                                 indices_forced=None, times_forced=None, durations_forced=None,
                                 max_episode=-1, photodiode_sampling=1000.,
                                 reverse_photodiodeSignal=False, verbose=False)
    nwbfile = pynwb.NWBFile(session_description='test', identifier='test',
                            session_start_time=datetime.datetime.now(tzlocal()))
    add_visual_stimulation(nwbfile, {'VisualStim': True}, None, args)
    with pynwb.NWBHDF5IO(str(tmp_path / 'test.nwb'), 'w') as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(str(tmp_path / 'test.nwb'), 'r') as io:
        data = types.SimpleNamespace(nwbfile=io.read())
        assert not any(is_vse_key(k) for k in data.nwbfile.stimulus)   # not an episode parameter
        vse = VisualStimMixin.get_visual_stim_vse(data, 0, 3)
        for key in ['t', 'x', 'y', 'max_amplitude']:
            np.testing.assert_allclose(vse[key], stim.vse[key])
        assert VisualStimMixin.get_visual_stim_vse(data, 0, 0) is None
        # the images are there too
        assert VisualStimMixin.get_visual_stim_image(data, 0, 1).shape == \
            tuple(stim.screen['resolution'][::-1])


VSE_DEG = {"seed": 3, "min-saccade-duration": 0.1, "max-saccade-duration": 1.0, "saccade-amplitude-deg": 30.}


def test_vse_in_degrees_same_angle_on_all_screens():
    """ saccade shifts in degrees -> the same visual angle on screens of different resolution """
    from physion.visual_stim.geometry import deg_per_pixel
    from physion.visual_stim.stimuli import virtual_scene_exploration as vse
    shifts = {}
    for screen in ['Dell-2020', 'Dell-2020-low-resolution']:
        stim = build_stim(dict(BASE, Stimulus='images-VSE', Screen=screen, NI_FOLDER=NI_BANK,
                               **dict(VSE_DEG, **{"Image-ID-1": "1.jpg"})))
        assert stim.vse['units'] == 'deg'
        t = stim.vse['t'][2] + 1e-3                     # after the 3rd saccade
        ix, iy = vse.compute_shifted_pixels(stim, t)
        dx, dz = deg_per_pixel(stim.screen, stim.units)
        shifts[screen] = (ix * dx, iy * dz)            # in degrees
        np.testing.assert_allclose(shifts[screen], (stim.vse['x'][2], stim.vse['y'][2]), atol=dx)
        stim.get_image(0, time_from_episode_start=t)   # no failure, even on the small screen


def test_vse_legacy_protocols_in_pixels():
    stim = build_stim(dict(BASE, Stimulus='grating-VSE', **VSE))  # "saccade-amplitude": pixels
    assert stim.vse['units'] == 'pixels' and stim.vse['x'].dtype.kind == 'i'
    from physion.visual_stim.build import get_default_params
    assert 'saccade-amplitude-deg' in get_default_params('images-VSE')


def test_vse_units_in_NWB(tmp_path):
    import datetime, types
    import pynwb
    from dateutil.tz import tzlocal
    from physion.assembling.nwb import add_visual_stimulation
    from physion.analysis.modalities.visual_stim import VisualStimMixin

    stim = build_stim(dict(BASE, Stimulus='grating-VSE', **VSE_DEG))
    add_vse_to_experiment(stim)
    np.save(tmp_path / 'visual-stim.npy', stim.experiment)
    args = types.SimpleNamespace(datafolder=str(tmp_path), modalities=['VisualStim'],
                                 force_to_visualStimTimestamps=True, exclude_episodes=[],
                                 indices_forced=None, times_forced=None, durations_forced=None,
                                 max_episode=-1, photodiode_sampling=1000.,
                                 reverse_photodiodeSignal=False, verbose=False)
    nwbfile = pynwb.NWBFile(session_description='test', identifier='test',
                            session_start_time=datetime.datetime.now(tzlocal()))
    add_visual_stimulation(nwbfile, {'VisualStim': True}, None, args)
    with pynwb.NWBHDF5IO(str(tmp_path / 'test.nwb'), 'w') as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(str(tmp_path / 'test.nwb'), 'r') as io:
        nwb = io.read()
        assert nwb.stimulus_template[vse_key(0, 3)].unit == 'degrees'
        vse = VisualStimMixin.get_visual_stim_vse(types.SimpleNamespace(nwbfile=nwb), 0, 3)
        assert vse['units'] == 'deg' and vse['max_amplitude'] == 30.
        np.testing.assert_allclose(vse['x'], stim.vse['x'])


def _write_NWB(stim, folder):
    """ visual-stim.npy with images and VSE -> NWB file, returns its path """
    import datetime, types
    import pynwb
    from dateutil.tz import tzlocal
    from physion.assembling.nwb import add_visual_stimulation

    add_images_to_experiment(stim)
    add_vse_to_experiment(stim)
    np.save(folder / 'visual-stim.npy', stim.experiment)
    args = types.SimpleNamespace(datafolder=str(folder), modalities=['VisualStim'],
                                 force_to_visualStimTimestamps=True, exclude_episodes=[],
                                 indices_forced=None, times_forced=None, durations_forced=None,
                                 max_episode=-1, photodiode_sampling=1000.,
                                 reverse_photodiodeSignal=False, verbose=False)
    nwbfile = pynwb.NWBFile(session_description='test', identifier='test',
                            session_start_time=datetime.datetime.now(tzlocal()))
    add_visual_stimulation(nwbfile, {'VisualStim': True}, None, args)
    with pynwb.NWBHDF5IO(str(folder / 'test.nwb'), 'w') as io:
        io.write(nwbfile)
    return str(folder / 'test.nwb')


def _rebuild_from_NWB(path, protocol):
    """ build_visual_stim, as when reading a recording, called on the opened NWB file """
    import json, types
    import pynwb
    from physion.analysis.modalities.visual_stim import VisualStimMixin

    class Data(VisualStimMixin):
        pass

    io = pynwb.NWBHDF5IO(path, 'r')
    data = Data()
    data.nwbfile, data.metadata = io.read(), json.loads(json.dumps(protocol))
    data.df_name = 'test'
    data.build_visual_stim()
    return data, io


def test_stored_vse_replaces_the_regenerated_one(tmp_path, capsys, monkeypatch):
    protocol = dict(BASE, Stimulus='grating-VSE', **VSE_DEG)
    stim = build_stim(json_copy(protocol))
    path = _write_NWB(stim, tmp_path)

    # same generator: no warning
    data, io = _rebuild_from_NWB(path, protocol)
    for key in ['t', 'x', 'y']:
        np.testing.assert_array_equal(data.visual_stim.vse[key], stim.vse[key])
    assert 'differs' not in capsys.readouterr().out
    io.close()

    # generator changed since the recording: warning, what was shown is kept
    from physion.visual_stim.stimuli import virtual_scene_exploration as vse
    original = vse.generate_sequence_from_protocol
    def changed(p):
        return dict(original(p), x=original(p)['x']+1.)
    monkeypatch.setattr(vse, 'generate_sequence_from_protocol', changed)
    data, io = _rebuild_from_NWB(path, protocol)
    for key in ['t', 'x', 'y']:
        np.testing.assert_array_equal(data.visual_stim.vse[key], stim.vse[key])
    assert 'differs' in capsys.readouterr().out
    assert data.visual_stim.vse['units'] == 'deg'
    io.close()


def json_copy(protocol):
    import json
    return json.loads(json.dumps(protocol))


def test_missing_image_files(tmp_path, capsys):
    import pytest
    missing = str(tmp_path / 'not-here')
    protocol = dict(BASE, Stimulus='images', **IMAGES, contrast=0.8)
    stim = build_stim(json_copy(protocol))
    path = _write_NWB(stim, tmp_path)
    protocol['NI_FOLDER'] = missing

    # building a protocol: the error is kept
    with pytest.raises(FileNotFoundError):
        build_stim(json_copy(protocol))

    # reading a recording: images of the NWB file, frames equal up to 8-bit rounding
    data, io = _rebuild_from_NWB(path, protocol)
    assert 'images-from-recording' not in data.metadata
    assert 'using the 3 images stored' in capsys.readouterr().out
    for ep in range(len(stim.experiment['index'])):
        np.testing.assert_allclose(data.visual_stim.get_image(ep), stim.get_image(ep),
                                   atol=0.8/255)
    io.close()


def test_missing_image_files_without_stored_images(tmp_path):
    import pytest
    missing = str(tmp_path / 'not-here')
    protocol = dict(BASE, Stimulus='images', **IMAGES)
    stim = build_stim(json_copy(protocol))
    # a file without images: it can still be read, the frames can not be drawn
    add_vse_to_experiment(stim)
    np.save(tmp_path / 'visual-stim.npy', stim.experiment)
    path = _write_NWB_without_images(stim, tmp_path)
    protocol['NI_FOLDER'] = missing
    data, io = _rebuild_from_NWB(path, protocol)
    with pytest.raises(FileNotFoundError):
        data.visual_stim.get_image(0)
    io.close()


def _write_NWB_without_images(stim, folder):
    path = _write_NWB(stim, folder)
    import h5py
    with h5py.File(path, 'a') as f:
        del f['stimulus/templates/visual-stim-images']
    return path


def test_missing_image_files_VSE_and_multiprotocol(tmp_path, capsys):
    """ images-VSE in a multiprotocol: stored VSE and images, shifted screen image with a warning """
    multi = {"Presentation": "multiprotocol", "Screen": "Dell-2020-low-resolution", "no-window": True,
             "presentation-prestim-period": 1, "presentation-poststim-period": 1}
    subs = [dict(BASE, Stimulus='grating'),
            dict(BASE, Stimulus='images-VSE', **IMAGES, **VSE_DEG)]
    for i, sub in enumerate(subs, start=1):
        multi['Protocol-%i' % i] = sub['Stimulus']
        for k, v in sub.items():
            multi['Protocol-%i-%s' % (i, k)] = v
    stim = build_stim(json_copy(multi))
    path = _write_NWB(stim, tmp_path)
    multi['Protocol-2-NI_FOLDER'] = str(tmp_path / 'not-here')

    data, io = _rebuild_from_NWB(path, multi)
    assert not any('images-from-recording' in k for k in data.metadata)
    rebuilt = data.visual_stim.STIM[1]
    assert rebuilt.NIarray is None and len(rebuilt.stored_images) == 3
    np.testing.assert_array_equal(rebuilt.vse['x'], stim.STIM[1].vse['x'])

    # before the first saccade: exact
    ep = 0
    np.testing.assert_allclose(rebuilt.get_image(ep, time_from_episode_start=0),
                               stim.STIM[1].get_image(ep, time_from_episode_start=0), atol=1./255)
    assert 'only approximated' not in capsys.readouterr().out
    # after a saccade: approximated, with a warning, same shape
    t = rebuilt.vse['t'][0]+1e-3
    assert rebuilt.get_image(ep, time_from_episode_start=t).shape == rebuilt.x.shape
    assert 'only approximated' in capsys.readouterr().out
    io.close()


FORMER_IDS = {"Image-ID-1": 0, "Image-ID-2": 3, "N-Image-ID": 2}


def _former_recording(protocol, tmp_path):
    """ NWB file of a former stimulus: images stored under the former "Image-ID" numbers """
    from physion.visual_stim.stimuli.images import FROM_RECORDING
    stim = build_stim(dict(json_copy(protocol), **{FROM_RECORDING: True}))
    rng = np.random.default_rng(0)
    shape = tuple(stim.screen['resolution'][::-1])
    stim.set_stored_images({i: rng.integers(0, 256, shape, dtype='uint8') for i in [0, 3]})
    return stim, _write_NWB(stim, tmp_path)


def test_former_natural_image_recording(tmp_path):
    protocol = dict(BASE, Stimulus='natural-image', NI_FOLDER='/not/here', contrast=0.8, radius=200,
                    **FORMER_IDS)
    stim, path = _former_recording(protocol, tmp_path)
    data, io = _rebuild_from_NWB(path, protocol)
    assert sorted(data.visual_stim.stored_images) == [0, 3]
    assert not any('images-from-recording' in k for k in data.metadata)
    assert sorted(set(data.visual_stim.experiment['Image-ID'])) == [0, 3]
    for ep in range(len(stim.experiment['index'])):
        np.testing.assert_allclose(data.visual_stim.get_image(ep), stim.get_image(ep), atol=1e-9)
    io.close()


def test_former_VSE_recordings_shifted_as_displayed(tmp_path):
    from physion.visual_stim.stimuli import virtual_scene_exploration as vse
    for i, (name, stimulus, amplitude) in enumerate([('deg', 'natural-image-VSE', VSE_DEG),
                                                     ('pixels', 'natural-image+VSE', dict(VSE, **{'saccade-amplitude': 20.}))]):
        folder = tmp_path / name
        folder.mkdir()
        protocol = dict(BASE, Stimulus=stimulus, contrast=1.0, radius=500, **FORMER_IDS, **amplitude)
        stim, path = _former_recording(protocol, folder)
        assert stim.vse['units'] == name[:3].replace('pix', 'pixels')
        data, io = _rebuild_from_NWB(path, protocol)
        for key in ['t', 'x', 'y']:
            np.testing.assert_array_equal(data.visual_stim.vse[key], stim.vse[key])
        t = stim.vse['t'][0]+1e-3
        expected = 0.5+1.0*(vse.compute_shifted_image(stim, stim.get_source_image(0), t)-0.5)
        ep = list(stim.experiment['Image-ID']).index(0)
        np.testing.assert_allclose(data.visual_stim.get_image(ep, time_from_episode_start=t),
                                   expected, atol=1e-9)
        io.close()


def test_former_set_of_images_recording(tmp_path):
    protocol = dict(BASE, Stimulus='set-of-images', **FORMER_IDS)
    stim, path = _former_recording(protocol, tmp_path)
    data, io = _rebuild_from_NWB(path, protocol)
    np.testing.assert_allclose(data.visual_stim.get_image(0), stim.get_image(0), atol=1e-9)
    io.close()


def test_former_recording_without_stored_images(tmp_path):
    import pytest
    protocol = dict(BASE, Stimulus='natural-image', **FORMER_IDS)
    stim, path = _former_recording(protocol, tmp_path)
    import h5py
    with h5py.File(path, 'a') as f:
        del f['stimulus/templates/visual-stim-images']
    data, io = _rebuild_from_NWB(path, protocol)
    assert len(data.visual_stim.experiment['Image-ID']) == len(stim.experiment['Image-ID'])
    with pytest.raises(FileNotFoundError):
        data.visual_stim.get_image(0)
    io.close()


def test_former_stimuli_can_not_build_protocols():
    import pytest
    for stimulus in ['natural-image', 'natural-image-VSE', 'natural-image+VSE', 'set-of-images']:
        with pytest.raises(ValueError, match='former recordings'):
            build_stim(dict(BASE, Stimulus=stimulus, **FORMER_IDS))


def test_read_tlim_without_acquisition():
    """ no acquisition: default time limits, no infinite loop """
    import types
    from physion.analysis.read_NWB import Data
    data = types.SimpleNamespace(nwbfile=types.SimpleNamespace(acquisition={}))
    Data.read_tlim(data)
    assert data.tlim == [0, 60*60]
