"""
visual stimulation geometry: pixel <-> angle mappings of the screens
"""
from types import SimpleNamespace
import numpy as np

from physion.visual_stim import geometry
from physion.visual_stim.screens import SCREENS


def V2_meshgrid(units):
    stim = SimpleNamespace(screen=SCREENS['LN-2screens'], units=units)
    geometry.set_angle_meshgrid_V2Screens(stim)
    return stim


def test_V2Screens_deg_units():
    """ 2-screens V-shaped setup in 'deg' units: angles in degrees, linear,
        same angle per pixel in both directions, spanning the altitudes of the screens """
    deg, cm = V2_meshgrid('deg'), V2_meshgrid('cm')
    for key in ['x', 'z', 'widths', 'heights', 'mask', 'screen_ids']:
        assert getattr(deg, key).shape == cm.x.shape
    dx = np.diff(deg.x[:, 0])
    dz = np.diff(deg.z[0, :])
    np.testing.assert_allclose(dx, -dz[0])          # linear, decreasing azimuth (as in 'cm')
    np.testing.assert_allclose(dz, dz[0])
    assert 0.05 < dz[0] < 0.2                        # ~0.08 deg per pixel (not radians)
    np.testing.assert_allclose([deg.z.min(), deg.z.max()], [cm.z.min(), cm.z.max()])
    assert 0.5 < deg.mask.mean() < 1.


def test_V2Screens_deg_units_consistent_with_cm_geometry():
    """ the screen point of each visible pixel of the angular grid has the angles of that pixel """
    screen = SCREENS['LN-2screens']
    L, lF = screen['width'], screen['distance_front']
    xMax = lF + np.abs(L - lF / np.sqrt(2)) / np.sqrt(2)
    deg = V2_meshgrid('deg')
    X, Z = deg.widths * xMax / L, deg.heights
    x, z = geometry.V2Screens_angles(X, Z, lF, np.where(deg.x >= 0, 1, 2))
    m = deg.mask
    np.testing.assert_allclose(np.degrees(x[m]), deg.x[m], atol=1e-8)
    np.testing.assert_allclose(np.degrees(z[m]), deg.z[m], atol=1e-8)
