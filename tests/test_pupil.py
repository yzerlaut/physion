"""
pupil tracking
"""
import os
import numpy as np
import pytest
from PyQt5 import QtWidgets

from physion.pupil.process import find_ellipse_props_of_binary_image_from_PCA


@pytest.mark.parametrize('sx, sy', [(20, 8), (6, 15), (10, 10.5)])
def test_ellipse_fit_on_binary_image(sx, sy):
    """ center and axes of a binary ellipse (numpy>=2: complex eigen values) """
    x, y = np.meshgrid(np.arange(100), np.arange(80), indexing='ij')
    cx, cy = 45, 38
    img = (((x-cx)/sx)**2 + ((y-cy)/sy)**2 <= 1).astype(int)

    mu, stds, angles = find_ellipse_props_of_binary_image_from_PCA(x, y, img)

    np.testing.assert_allclose(mu, (cx, cy), atol=1)
    # for a uniform ellipse, the std along an axis is (semi-axis)/2
    np.testing.assert_allclose(sorted(stds), sorted([2*sx, 2*sy]), rtol=0.1)
    assert np.all(np.isreal(angles))


SESSION = os.environ.get('PHYSION_TEST_PUPIL_FOLDER',
            os.path.expanduser('~/DATA/physion_Demo-Datasets/PYR-WT/processed/2025_11_14/13-54-32'))


@pytest.mark.skipif(not os.path.isfile(os.path.join(SESSION, 'pupil.npy')),
                    reason='no pupil data at "%s"' % SESSION)
def test_pupil_window_shortcuts(gui, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: SESSION))
    gui.data = loaded = object()

    window = gui.pupil()
    gui.open()                          # [O]
    assert len(window.pupilData['frame'])>0

    window.cframe = 100
    gui.refresh()                       # [R]
    assert window.img is not None
    gui.press1()                        # [1]
    window.cframe = 200
    gui.press2()                        # [2]
    assert (window.cframe1, window.cframe2) == (100, 200)

    gui.fit()                           # [F]
    assert np.all(np.isfinite(window.fit_pupil(coords_only=True)))

    sx = window.pupilData['sx'].copy()
    gui.press5()                        # [5] exclude outliers
    gui.press5()                        #     and back
    np.testing.assert_array_equal(window.pupilData['sx'], sx)

    assert gui.data is loaded
    assert gui.slot_errors == []
