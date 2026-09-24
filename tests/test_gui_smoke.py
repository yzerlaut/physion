"""
headless GUI tests: every window opens, and windows do not interfere
"""
import os
import numpy as np
import h5py
import pytest
from PyQt5 import QtWidgets


def menu_actions(gui):
    for menu in [gui.fileMenu, gui.experimentMenu, gui.preprocessingMenu,
                 gui.assemblingMenu, gui.visualizationMenu,
                 gui.analysisMenu, gui.otherMenu]:
        for action in menu.actions():
            if 'Quit' not in action.text():
                yield '%s > %s' % (menu.title().strip(), action.text().strip()), action


def test_all_menu_actions_open_without_data(gui, qapp):
    failures = []
    for name, action in menu_actions(gui):
        gui.data = None
        gui.slot_errors.clear()
        action.trigger()
        qapp.processEvents()
        if gui.slot_errors:
            failures.append('%s -> %r' % (name, gui.slot_errors[0]))
    assert failures == []


@pytest.mark.parametrize('window', ['pupil', 'facemotion', 'bot_spatial_maps',
                                    'intrinsic', 'SS_intrinsic', 'OD_analysis',
                                    'h5_imaging_UI', 'suite2p_preprocessing_UI'])
def test_windows_keep_the_loaded_NWB(gui, window):
    """ a window opened in another tab should not reset the loaded data """
    loaded = object() # stands for a physion.analysis.read_NWB.Data
    gui.data = loaded
    getattr(gui, window)()
    assert gui.data is loaded
    assert gui.slot_errors == []


def test_h5_imaging_UI(gui, tmp_path, monkeypatch):
    T, Ny, Nx = 300, 40, 50
    movie = np.ones((T, Ny, Nx), dtype=np.float32)
    signal = 5*np.sin(np.arange(T)/10.)
    movie[:, 10:20, 10:20] += signal[:, None, None]
    fn = str(tmp_path/'movie.h5')
    with h5py.File(fn, 'w') as f:
        f['data'] = movie
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (fn, '')))

    gui.h5_imaging_UI()
    gui.open_h5_imaging()
    np.testing.assert_allclose(gui.h5MeanImg, movie[:100].mean(axis=0))

    gui.h5ExpBox.setValue(0.5) # display exponent -> redraw
    norm = gui.h5MeanImg-gui.h5MeanImg.min()
    np.testing.assert_allclose(gui.h5Img.image, (norm/norm.max())**0.5)

    gui.add_ROI_h5()
    roi = gui.h5ROIs[0]
    roi.setPos((11, 11)); roi.setSize((8, 8))
    gui.extract_fluo_h5()
    np.testing.assert_allclose(gui.h5Fluo[0], 1+signal, rtol=1e-5)

    gui.save_ROIs_h5()
    saved = np.load(str(tmp_path/'movie_ROIs.npy'), allow_pickle=True).item()
    assert saved['fluorescence'].shape == (1, T)
    assert gui.slot_errors == []
