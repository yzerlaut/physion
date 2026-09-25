"""
trial-averaging window
"""
import os, glob, types
import numpy as np
import pytest

from physion.analysis.trial_averaging import TrialAveragingWindow


def test_conditions_follow_the_built_episodes():
    """ episodes can be missing (e.g. outside the recording): the conditions
        are built from the parameters of the episodes that were built """
    angles = np.array([0., 90., 0., 90., 0., 90.])
    kept = np.array([0, 1, 3, 4, 5])            # episode 2 is missing
    EPISODES = types.SimpleNamespace(time_start=np.arange(6)[kept], angle=angles[kept],
                                     varied_parameters={'angle': np.array([0., 90.])})
    box = types.SimpleNamespace(currentText=lambda: 'angle         (column)')
    window = types.SimpleNamespace(EPISODES=EPISODES, box0=box)
    window.build_conditions = lambda option: TrialAveragingWindow.build_conditions(window, option)

    columns = TrialAveragingWindow.build_column_conditions(window)
    np.testing.assert_array_equal(columns[0], angles[kept]==0.)
    np.testing.assert_array_equal(columns[1], angles[kept]==90.)
    # no parameter set to "(row)" -> a single row with all the episodes
    rows = TrialAveragingWindow.build_row_conditions(window)
    assert len(rows)==1 and rows[0].all()


NWB = os.environ.get('PHYSION_TEST_NWB_TUNING',
        (sorted(glob.glob(os.path.expanduser(
            '~/DATA/Taddy/PN_shGrid1-2026/NWBs/Orientation-Contrast/*.nwb')))+[''])[0])


@pytest.mark.skipif(not os.path.isfile(NWB), reason='no orientation-tuning NWB file')
def test_trial_averaging_window(gui):
    from physion.analysis.read_NWB import Data
    gui.data = Data(NWB, verbose=False)
    window = gui.trial_averaging()
    quantities = [window.qbox.itemText(i) for i in range(window.qbox.count())]
    assert 'dFoF' in quantities

    window.pbox.setCurrentIndex(2)             # "tuning-low-contrast"
    window.update_protocol_TA()
    window.qbox.setCurrentIndex(quantities.index('dFoF'))
    window.box0.setCurrentIndex(3)             # angle -> (column)
    window.roiPickTA.setText('3')
    window.compute_episodes()
    gui.refresh()                              # [R]

    ep = window.EPISODES
    for i, angle in enumerate(ep.varied_parameters['angle']):
        plotted = window.AX[0][i].listDataItems()[-1].getData()[1]
        np.testing.assert_allclose(plotted, ep.dFoF[ep.angle==angle][:, 3, :].mean(axis=0))
    gui.fit()                                  # [F]
    assert gui.slot_errors == []
