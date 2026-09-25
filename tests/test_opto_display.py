"""
optogenetics in the raw-data window: blue overlay when the opto is on
"""
import os, shutil, types
import numpy as np
import pytest

from physion.dataviz.plots import opto_periods


@pytest.mark.parametrize('signal, periods', [
    ([0, 1, 1, 0, 0, 1, 0], [(1, 3), (5, 6)]),
    ([1, 1, 0, 0, 1, 1, 1], [(0, 2), (4, 6)]),   # on at the start / at the end
    ([0, 0, 0], []),
])
def test_opto_periods(signal, periods):
    data = types.SimpleNamespace(opto=np.array(signal), t_opto=np.arange(len(signal))*1.)
    data.build_opto = lambda verbose=False: None
    t_on, t_off = opto_periods(data)
    assert list(zip(t_on, t_off)) == periods


DEMO = os.path.expanduser('~/DATA/physion_Demo-Datasets/PYR-WT/2025_11_14-13-54-32.nwb')


@pytest.mark.skipif(not os.path.isfile(DEMO), reason='no demo NWB file')
def test_opto_overlay(gui, tmp_path, monkeypatch):
    import pynwb
    from pynwb.ogen import OptogeneticStimulusSite, OptogeneticSeries
    from PyQt5 import QtWidgets
    from physion.dataviz.tools import settings

    # the demo recording + an optogenetic stimulation (on: 10-12s, 20-25s)
    fn = str(tmp_path/'opto.nwb')
    shutil.copy(DEMO, fn)
    with pynwb.NWBHDF5IO(fn, 'a') as io:
        nwb = io.read()
        site = OptogeneticStimulusSite(name='OptogeneticStimulusSite', description=' ',
                        device=nwb.create_device(name='LED'), excitation_lambda=470., location='VIS')
        nwb.add_ogen_site(site)
        t = np.arange(0, 60, 1e-3)
        on = ((t>=10) & (t<12)) | ((t>=20) & (t<25))
        nwb.add_stimulus(OptogeneticSeries(name='OptogeneticSeries', data=on.astype(int),
                                           site=site, rate=1000.))
        io.write(nwb)

    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (fn, '')))
    gui.open()                                   # [O] -> raw data window
    window = gui.window_objects[gui.tabWidget.currentIndex()]
    assert window.optoSelect.isEnabled() and window.optoSelect.isChecked()

    blue = tuple(settings['colors']['Opto'])
    def bands():
        return [np.round(item.getData()[0], 3).tolist() for item in window.plot.listDataItems()
                if tuple(np.atleast_1d(item.opts.get('fillBrush') or ())) == blue]
    window.raw_data_plot([0, 30])
    assert bands() == [[10., 12.], [20., 25.]]
    window.optoSelect.setChecked(False)
    window.raw_data_plot([0, 30])
    assert bands() == []
    assert gui.slot_errors == []
