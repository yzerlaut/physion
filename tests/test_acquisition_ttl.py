"""
TTL of the visual stimulation on the digital outputs of the NIdaq card
"""
import numpy as np

from physion.acquisition.tools import visual_stim_TTL_steps

EXPERIMENT = {'time_start': [1., 4., 7.5], 'time_stop': [3., 6., 9.5]}


def test_TTL_lasts_the_whole_episode():
    steps = visual_stim_TTL_steps(EXPERIMENT, 2)
    assert [s['channel'] for s in steps] == [2, 2, 2]
    assert [s['onset'] for s in steps] == EXPERIMENT['time_start']
    assert [s['onset']+s['duration'] for s in steps] == EXPERIMENT['time_stop']


def test_TTL_from_a_built_protocol():
    from physion.visual_stim.build import build_stim
    stim = build_stim({"Presentation": "Stimuli-Sequence", "Stimulus": "grating",
                       "Screen": "Dell-2020-low-resolution", "presentation-duration": 2,
                       "presentation-prestim-period": 1, "presentation-poststim-period": 1,
                       "presentation-interstim-period": 3, "presentation-interstim-jitter": 1,
                       "N-repeat": 3})
    for step, duration in zip(visual_stim_TTL_steps(stim.experiment, 1),
                              stim.experiment['time_duration']):
        np.testing.assert_allclose(step['duration'], duration)
