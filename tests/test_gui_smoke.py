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
    folders = []
    def dialog(parent, title, folder, **kwargs):
        folders.append(folder)
        return (fn, '')
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', staticmethod(dialog))

    window = gui.h5_imaging_UI()
    gui.open() # [O] shortcut -> handled by the window of the current tab
    np.testing.assert_allclose(window.meanImg, movie[:100].mean(axis=0))
    assert folders == [window.choose_root_folder()] # its own folder box

    window.expBox.setValue(0.5) # display exponent -> redraw
    norm = window.meanImg-window.meanImg.min()
    np.testing.assert_allclose(window.img.image, (norm/norm.max())**0.5)

    window.add_ROI()
    roi = window.ROIs[0]
    roi.setPos((11, 11)); roi.setSize((8, 8))
    window.extract_fluo()
    np.testing.assert_allclose(window.fluo[0], 1+signal, rtol=1e-5)

    gui.save() # [S] shortcut
    saved = np.load(str(tmp_path/'movie_ROIs.npy'), allow_pickle=True).item()
    assert saved['fluorescence'].shape == (1, T)
    assert gui.slot_errors == []


def test_shortcuts_ignore_a_replaced_window(gui):
    """ a window replaced by another one in the same tab gets no shortcut """
    from physion.gui.window import current_window
    window = gui.h5_imaging_UI(tab_id=2)
    assert current_window(gui) is window
    # a window built from functions only sets its name (see physion.gui.parts)
    gui.cleanup_tab(gui.tabs[2])
    gui.windows[2] = 'function-based window'
    assert current_window(gui) is None


@pytest.mark.parametrize('window', ['h5_imaging_UI', 'pupil', 'facemotion', 'FOV_coords_UI',
        'intrinsic', 'SS_intrinsic', 'OD_analysis', 'bot_spatial_maps', 'suite2p_preprocessing_UI',
        'red_channel_labelling', 'spike_sorting_preprocessing_UI', 'build_DataTable_UI',
        'build_NWB_from_DataTable_UI', 'build_NWB_UI', 'transfer_gui', 'cameraData_to_movie_gui',
        'imaging_to_movie_gui', 'deletion_gui'])
def test_shortcut_handlers_are_not_shadowed(gui, window):
    """ the "on_*" shortcut handlers are not hidden by window attributes """
    from physion.gui.window import Window, SHORTCUTS
    obj = getattr(gui, window)()
    assert isinstance(obj, Window)
    for handler in SHORTCUTS:
        if hasattr(type(obj), handler):
            assert callable(getattr(obj, handler)), handler
