"""
finding the matching imaging FOV across days (physion.imaging.matching_FOV_gui)
"""
import os, time
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter
from PyQt5 import QtWidgets

from physion.imaging.matching_FOV_gui import shifted_correlation, preprocess,\
        find_sample_images, load_image, channel_of


def fov(seed=0, shape=(128, 160)):
    """ an imaging FOV: cells/vessels-like structure + slow illumination """
    rng = np.random.default_rng(seed)
    big = (np.array(shape)*1.6).astype(int)
    texture = gaussian_filter(rng.normal(0, 1, big), 2)+\
              gaussian_filter(rng.normal(0, 1, big), 6)
    return texture


def view(texture, x0, y0, shape=(128, 160), seed=1, noise=0.05):
    """ the FOV at position (x0, y0), with vignetting and pixel noise """
    rng = np.random.default_rng(seed)
    img = texture[y0:y0+shape[0], x0:x0+shape[1]].copy()
    img = img*(0.4+0.6*np.linspace(0, 1, shape[1]))[np.newaxis,:]
    return img+noise*rng.normal(0, 1, shape)+3.


@pytest.mark.parametrize('dx, dy', [(0, 0), (13, -7), (-30, 25), (38, 30)])
def test_shifted_correlation_finds_the_shift(dx, dy):
    texture = fov()
    ref = view(texture, 50, 40, seed=1)
    img = view(texture, 50-dx, 40-dy, seed=2)  # ref pixel p at p+(dx, dy) in img
    r, sdx, sdy = shifted_correlation(preprocess(ref), preprocess(img))
    assert (sdx, sdy) == (dx, dy)
    assert r > 0.9


def test_shifted_correlation_limited_to_25_percent():
    texture = fov()
    ref = view(texture, 10, 10)
    img = view(texture, 10+70, 10)  # shift of 70 px > 25% of 160 px
    r, dx, dy = shifted_correlation(preprocess(ref), preprocess(img))
    assert abs(dx) <= 40 and abs(dy) <= 32
    assert r < 0.5
    # another FOV: low correlation
    other = view(fov(seed=5), 10, 10)
    assert shifted_correlation(preprocess(ref), preprocess(other))[0] < 0.5


def test_load_image_png_and_tif16(tmp_path):
    a = (np.arange(12*10).reshape(12, 10)*500).astype(np.uint16)
    Image.fromarray(a).save(tmp_path/'img.tif')
    np.testing.assert_array_equal(load_image(tmp_path/'img.tif'), a)
    rgba = np.zeros((12, 10, 4), dtype=np.uint8); rgba[..., :3] = 100; rgba[..., 3] = 255
    Image.fromarray(rgba).save(tmp_path/'img.png')
    np.testing.assert_allclose(load_image(tmp_path/'img.png'), 100, atol=1)


def bruker_sample(folder, name, image, channel='Ch2'):
    """ a Bruker "SingleImage-xxx" folder (with its "References" copies) """
    os.makedirs(os.path.join(folder, name, 'References'))
    for ch in ['Ch1', channel]:
        Image.fromarray(image.astype(np.float32)).save(
            os.path.join(folder, name, '%s_Cycle00001_%s_000001.ome.tif' % (name, ch)))
    Image.fromarray(image.astype(np.float32)).save(
        os.path.join(folder, name, 'References', '%s-%s-16bit-Reference.tif' % (name, channel)))


def test_find_sample_images(tmp_path):
    for i in range(3):
        bruker_sample(tmp_path, 'SingleImage-001-%i' % i, np.ones((4, 4)))
        time.sleep(0.01)
    files = find_sample_images(tmp_path, channel='Ch2')
    assert [os.path.basename(os.path.dirname(f)) for f in files] ==\
                ['SingleImage-001-0', 'SingleImage-001-1', 'SingleImage-001-2']
    assert all(channel_of(f)=='Ch2' and 'References' not in f for f in files)
    assert len(find_sample_images(tmp_path)) == 6   # no channel: both channels


def test_matching_FOV_UI(gui, tmp_path, monkeypatch):
    texture = fov()
    ref = view(texture, 50, 40)
    Image.fromarray(ref.astype(np.float32)).save(tmp_path/'ref_Ch2.tif')
    scan = tmp_path/'scan'
    scan.mkdir()
    shifts = [(30, 0), (5, -3), (0, 25)]
    for i, (dx, dy) in enumerate(shifts):
        bruker_sample(scan, 'SingleImage-%i' % i, view(texture, 50-dx, 40-dy, seed=i+3))
        time.sleep(0.01)
    # a sample of another FOV
    bruker_sample(scan, 'SingleImage-3', view(fov(seed=7), 50, 40))
    time.sleep(0.01)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (str(tmp_path/'ref_Ch2.tif'), '')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: str(scan)))

    window = gui.matching_FOV_UI()
    gui.open()                       # [O] -> load ref. image
    assert window.ref.shape == ref.shape
    window.scanBtn.click()           # select scan folder (+ first scan)
    window.scan()                    # files written: added at the next scan
    assert len(window.samples) == 4  # one channel (Ch2), no "References"
    assert [(s['dx'], s['dy']) for s in window.samples[:3]] == shifts
    assert all(s['r'] > 0.9 for s in window.samples[:3])
    assert window.samples[3]['r'] < 0.5   # another FOV
    assert window.current == 3        # the last sample is displayed

    # a new sample written during the search, at the reference position
    bruker_sample(scan, 'SingleImage-4', view(texture, 50, 40, seed=10))
    window.scan(); window.scan()
    assert len(window.samples) == 5
    assert (window.samples[4]['dx'], window.samples[4]['dy']) == (0, 0)
    best = window.best_sample()
    assert window.samples[best]['r'] == max(s['r'] for s in window.samples)
    assert 'best: #%i' % (best+1) in window.plot.titleLabel.text

    # display exponent, and a sample chosen on the plot
    window.expBox.setValue(0.3)
    window.show_sample(0)
    assert window.sampleFOV.isVisible()
    assert window.sampleFOV.rect().x() == -30

    # a new reference: correlations computed again
    gui.open()
    assert all(s['r'] is not None for s in window.samples)
    assert gui.slot_errors == []

    # "update": correlations of the images with the exponent transformation
    from physion.imaging.matching_FOV_gui import display_image
    raw = [s['r'] for s in window.samples]
    window.expBox.setValue(0.4)
    window.updateBtn.click()
    assert window.analysis_exponent == 0.4
    assert '0.40' in window.analysisLabel.text()
    r, dx, dy = shifted_correlation(preprocess(display_image(window.ref, 0.4)),
                    preprocess(display_image(window.samples[0]['image'], 0.4)))
    assert window.samples[0]['r'] == pytest.approx(r)
    assert window.samples[0]['r'] != pytest.approx(raw[0])
    assert (window.samples[0]['dx'], window.samples[0]['dy']) == (dx, dy)
    # the next samples: same transformation
    bruker_sample(scan, 'SingleImage-5', view(texture, 45, 40, seed=11))
    window.scan(); window.scan()
    r = shifted_correlation(preprocess(display_image(window.ref, 0.4)),
                    preprocess(display_image(window.samples[5]['image'], 0.4)))[0]
    assert window.samples[5]['r'] == pytest.approx(r)
    assert gui.slot_errors == []

    # replaced by another window in its tab: no more scan
    gui.h5_imaging_UI(tab_id=window.tab_id)
    window.scan()
    assert not window.timer.isActive()
