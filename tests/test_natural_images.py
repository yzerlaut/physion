"""
natural images: morphing, spectrum equalization, sparseness, calibration
    (synthetic images, no image bank needed)
"""
import numpy as np
import pytest

from physion.visual_stim import geometry
from physion.visual_stim.screens import SCREENS
from physion.visual_stim.natural_images import images, spectrum, morphing, \
    mouse_vision, sparseness, calibration

SHAPE = (96, 128)
DEG_PER_PX = 0.5          # small images covering a mouse-relevant field (48x64 deg)


def edges_image(seed):
    """ sparse image: random overlapping rectangles on a gray background """
    rng = np.random.default_rng(seed)
    img = np.full(SHAPE, 0.5)
    for _ in range(12):
        y, x = rng.integers(0, SHAPE[0] - 10), rng.integers(0, SHAPE[1] - 10)
        h, w = rng.integers(5, 40), rng.integers(5, 40)
        img[y:y + h, x:x + w] = rng.uniform()
    return img


@pytest.fixture
def imgs():
    return np.array([edges_image(seed) for seed in range(4)])


def test_periodic_smooth_decomposition():
    u = edges_image(0)
    p, s = spectrum.periodic_smooth(u)
    np.testing.assert_allclose(p + s, u, atol=1e-10)
    # the smooth part carries the boundary discontinuity: the periodic part wraps smoothly
    assert np.abs(p[0] - p[-1]).mean() < np.abs(u[0] - u[-1]).mean()


def test_normalize():
    x = images.normalize(edges_image(1), mean_lum=0.5, rms_contrast=0.1)
    assert x.min() >= 0 and x.max() <= 1
    assert abs(x.mean() - 0.5) < 0.01 and abs(x.std() - 0.1) < 0.01


def test_phase_morph_endpoints(imgs):
    morpher = morphing.PhaseMorpher(imgs)
    # same image -> the morph is the (normalized) image itself, whatever w
    np.testing.assert_allclose(morpher.morph(0, 0, 0.3), images.normalize(imgs[0]), atol=1e-8)
    # w=0 / w=1 keep the phases of A / B
    for w, k in ((0., 0), (1., 1)):
        x = morpher.morph(0, 1, w)
        assert mouse_vision.corr(x, imgs[k]) > mouse_vision.corr(x, imgs[1 - k])


def test_pair_weights_and_pairs(imgs):
    assert morphing.pair_weights(0.) == (0., 1.)
    assert morphing.pair_weights(1.) == (0.5, 0.5)
    assert morphing.pair_weights(1., anchored=True) == (0., 0.)
    morpher = morphing.PhaseMorpher(imgs)
    x1, x2 = morpher.pair(0, 1, 1.)
    np.testing.assert_allclose(x1, x2)
    r = [mouse_vision.corr(*morpher.pair(0, 1, s)) for s in (0., 0.5, 0.9)]
    assert r[0] < r[1] < r[2]


@pytest.mark.parametrize('mode', ['radial', 'full'])
def test_equalize_pair_spectra(mode):
    x1, x2 = spectrum.equalize_pair(edges_image(0), edges_image(1), mode=mode, rms_contrast=0.1)
    bins = spectrum.radial_bins(SHAPE)
    radial = [np.bincount(bins.ravel(), np.abs(np.fft.fft2(spectrum.periodic_smooth(x - x.mean())[0])).ravel())
              for x in (x1, x2)]
    # same spectral shape: the overall scale differs as each image (including its own smooth
    # component, not equalized) is then set to the same RMS contrast
    np.testing.assert_allclose(radial[0] / radial[0].sum(), radial[1] / radial[1].sum(), rtol=0.1)


def test_kurtosis():
    rng = np.random.default_rng(0)
    assert abs(sparseness.kurtosis(rng.standard_normal(100000)) - 3) < 0.1
    assert sparseness.kurtosis(rng.laplace(size=100000)) > 5


def test_naturalness_of_continuum(imgs):
    model = sparseness.SparsenessModel(SHAPE, deg_per_px=DEG_PER_PX, downsampling=1)
    ws = np.linspace(0, 1, 5)
    res = sparseness.continuum_sparseness(imgs[0], imgs[1], ws, model)
    assert res['K'].shape == (5, len(model.sfs) * model.n_ori)
    assert model.per_sf(res['K']).shape == (5, len(model.sfs))
    # sparse originals, Gaussian scrambled control
    assert res['K_orig'].mean() > res['K_scr'].mean() + 1
    assert abs(res['K_scr'].mean() - 3) < 1
    # the midpoint is less natural than the ends of the continuum
    assert res['naturalness'][2] < min(res['naturalness'][0], res['naturalness'][-1])


def test_calibration(imgs):
    morpher = morphing.PhaseMorpher(imgs)
    v1 = mouse_vision.build_v1_model([morpher.equalized(i) for i in range(len(imgs))],
                                     deg_per_px=DEG_PER_PX, downsampling=1)
    s_grid = np.linspace(0, 1, 6)
    pairs = calibration.random_pairs(len(imgs), 3)
    calib, v1_norm, calibrated_s = calibration.calibrate(morpher.pair, pairs, s_grid, v1,
                                                         verbose=False)
    assert set(calib) == set(mouse_vision.METRICS)
    assert np.all(np.diff(v1_norm) > 0)
    assert calibrated_s(0.) == pytest.approx(0.) and calibrated_s(1.) == pytest.approx(1.)


def test_deg_per_pixel_single_screen():
    screen = SCREENS['Dell-2020']
    # 'deg' units: pixels linear in angle, the screen height spans 2*arctan(H/2/d)
    expected = np.degrees(2 * np.arctan(screen['height'] / 2 / screen['distance_from_eye'])) \
        / (screen['resolution'][1] - 1)
    dx, dz = geometry.deg_per_pixel('Dell-2020', units='deg')
    assert dx == pytest.approx(expected, rel=0.02) and dz == pytest.approx(expected, rel=0.02)
    # 'cm' units: pixels linear on the screen, the angle per pixel is largest at the center
    dx, dz = geometry.deg_per_pixel('Dell-2020', units='cm')
    w_px = screen['width'] / screen['resolution'][0]
    assert dx == pytest.approx(np.degrees(np.arctan(w_px / screen['distance_from_eye'])), rel=0.01)
    assert dx > geometry.deg_per_pixel('Dell-2020', units='cm', where='mean')[0]
    # image stretched over the screen: scaled by the screen pixels per image pixel
    ix, iz = geometry.deg_per_pixel('Dell-2020', units='cm', image_shape=(72, 128))
    assert (ix, iz) == pytest.approx((10 * dx, 10 * dz))


def test_models_from_screen_key():
    shape = (180, 320)                      # same aspect ratio as the Dell-2020 screen
    expected = np.mean(geometry.deg_per_pixel('Dell-2020', 'cm', image_shape=shape))
    assert mouse_vision.image_deg_per_px(shape, 'Dell-2020', 'cm') == pytest.approx(expected)
    model = sparseness.SparsenessModel(shape, screen='Dell-2020', units='cm')
    assert model.deg_per_px == pytest.approx(expected)
    # explicit geometry overrides the screen
    assert sparseness.SparsenessModel(shape, deg_per_px=0.3).deg_per_px == 0.3
    # non-square image pixels on the screen -> warning
    with pytest.warns(UserWarning):
        mouse_vision.image_deg_per_px((180, 180), 'Dell-2020', 'cm')
