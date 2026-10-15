"""
images and virtual saccadic explorations (VSE) of the protocols stored in visual-stim.npy
    ("protocol_id-%i-Image-ID-%i" and "protocol_id-%i-VSE-%i" keys) and in the NWB files
"""
import numpy as np

from physion.visual_stim.build import build_stim, add_images_to_experiment, \
    image_key, is_image_key, add_vse_to_experiment, vse_key, is_vse_key

BASE = {"Presentation": "Stimuli-Sequence", "Screen": "Dell-2020-low-resolution",
        "no-window": True, "presentation-duration": 1, "presentation-prestim-period": 1,
        "presentation-poststim-period": 1, "presentation-interstim-period": 1, "N-repeat": 2}


def test_image_keys():
    assert image_key(3, 0) == 'protocol_id-3-Image-ID-0'
    assert is_image_key(image_key(3, 0))
    assert not is_image_key('protocol_id') and not is_image_key('Image-ID')


def test_natural_images_in_experiment():
    stim = build_stim(dict(BASE, Stimulus='natural-image',
                           **{"Image-ID-1": 0, "Image-ID-2": 2, "N-Image-ID": 3, "contrast": 0.8}))
    add_images_to_experiment(stim)
    keys = sorted(k for k in stim.experiment if is_image_key(k))
    assert keys == [image_key(0, i) for i in range(3)]
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
                   (2, dict(BASE, Stimulus='natural-image', **{"Image-ID-1": 1, "Image-ID-2": 2, "N-Image-ID": 2}))]:
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

    # a recording folder with the visual-stim.npy of a natural-image protocol
    stim = build_stim(dict(BASE, Stimulus='natural-image',
                           **{"Image-ID-1": 0, "Image-ID-2": 2, "N-Image-ID": 3}))
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
        for image_id in range(3):
            img = VisualStimMixin.get_visual_stim_image(data, 0, image_id)
            np.testing.assert_array_equal(img, stim.experiment[image_key(0, image_id)])
        assert VisualStimMixin.get_visual_stim_image(data, 0, 5) is None
        assert VisualStimMixin.get_visual_stim_image(data, 1, 0) is None


VSE = {"seed": 3, "min-saccade-duration": 0.1, "max-saccade-duration": 1.0, "saccade-amplitude": 200.}


def test_vse_keys():
    assert vse_key(3, 0) == 'protocol_id-3-VSE-0'
    assert is_vse_key(vse_key(3, 0)) and not is_image_key(vse_key(3, 0))
    assert not is_vse_key(image_key(3, 0))


def test_vse_in_experiment():
    """ natural-image-VSE and grating-VSE in a multiprotocol, with a grating (no VSE) """
    multi = {"Presentation": "multiprotocol", "Screen": "Dell-2020-low-resolution", "no-window": True,
             "presentation-prestim-period": 1, "presentation-poststim-period": 1}
    subs = [dict(BASE, Stimulus='grating'),
            dict(BASE, Stimulus='natural-image-VSE', **VSE),
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

    stim = build_stim(dict(BASE, Stimulus='natural-image-VSE',
                           **dict(VSE, **{"Image-ID-1": 0, "Image-ID-2": 1, "N-Image-ID": 2})))
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
        stim = build_stim(dict(BASE, Stimulus='natural-image-VSE', Screen=screen,
                               **dict(VSE_DEG, **{"Image-ID-1": 0, "Image-ID-2": 0, "N-Image-ID": 1})))
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
    assert 'saccade-amplitude-deg' in get_default_params('natural-image-VSE')


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
