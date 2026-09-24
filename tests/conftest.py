"""
shared fixtures for the automated tests (run with: `pytest` from the repo root)

    - synthetic Bruker TSeries folders (xml + single-page tiffs)
    - a headless main window (Qt "offscreen" platform)
"""
import os, sys, pathlib
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from PIL import Image

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen') # no display needed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))


###########################################################################
####           synthetic Bruker TSeries                               #####
###########################################################################

def write_TSeries(folder, name='TSeries-01012025-001',
                  nframes=40, channels=('Ch1 Red', 'Ch2 Green'),
                  nplanes=1, shape=(12, 16), seed=0):
    """
    writes a Bruker-like TSeries folder (xml with Prairie 5.5 layout + tiffs)

    returns the folder path and the movies: {(channel, plane): (nframes, Ly, Lx)}
    """
    TS = os.path.join(folder, name)
    os.makedirs(TS)
    rng = np.random.default_rng(seed)
    movies = {(c, p): rng.integers(0, 8000, (nframes,)+shape, dtype=np.uint16)\
                    for c in channels for p in range(nplanes)}

    root = ET.Element('PVScan', version='5.5.64.500',
                      date='24-Mar-25 10:51:35 AM', notes='')
    ET.SubElement(root, 'SystemIDs')
    settings = ET.SubElement(root, 'PVStateShard')
    ET.SubElement(settings, 'PVStateValue', key='framePeriod', value='0.03')
    mpp = ET.SubElement(settings, 'PVStateValue', key='micronsPerPixel')
    for axis in ['XAxis', 'YAxis']:
        ET.SubElement(mpp, 'IndexedValue', index=axis, value='1.5')
    laser = ET.SubElement(settings, 'PVStateValue', key='laserWavelength')
    ET.SubElement(laser, 'IndexedValue', index='0', value='920',
                  description='Laser')

    # single plane: one sequence of frames
    # multi-plane: one sequence (cycle) per time point, one frame per plane
    if nplanes==1:
        cycles = [[(t, 0) for t in range(nframes)]]
    else:
        cycles = [[(t, p) for p in range(nplanes)] for t in range(nframes)]

    for icycle, cycle in enumerate(cycles):
        seq = ET.SubElement(root, 'Sequence', type='TSeries Timed Element',
                            cycle=str(icycle+1), time='10:51:35.2747495')
        for i, (t, p) in enumerate(cycle):
            frame = ET.SubElement(seq, 'Frame',
                                  relativeTime=str(0.03*t),
                                  absoluteTime=str(10+0.03*t),
                                  index=str(i+1 if nplanes==1 else p+1))
            for ic, c in enumerate(channels):
                fn = '%s_Cycle%05i_Ch%i_%06i.ome.tif' % (name, icycle+1, ic+1,
                                                        i+1 if nplanes==1 else p+1)
                ET.SubElement(frame, 'File', channel=str(ic+1),
                              channelName=c, filename=fn)
                Image.fromarray(movies[(c, p)][t]).save(os.path.join(TS, fn))

    ET.ElementTree(root).write(os.path.join(TS, name+'.xml'))
    return TS, movies


@pytest.fixture
def make_TSeries(tmp_path):
    """ factory: make_TSeries(nframes=.., channels=.., nplanes=..) """
    def factory(**kwargs):
        return write_TSeries(str(tmp_path), **kwargs)
    return factory


###########################################################################
####           headless GUI                                           #####
###########################################################################

@pytest.fixture(scope='session')
def qapp():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def gui(qapp, monkeypatch):
    """
    main window with:
        - file dialogs returning "cancelled" (overwrite them in tests if needed)
        - exceptions raised in Qt slots collected in `gui.slot_errors`
          (Qt does not propagate them to the caller of `action.trigger()`)
    """
    from PyQt5 import QtWidgets
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: ''))
    errors = []
    monkeypatch.setattr(sys, 'excepthook',
                        lambda t, v, tb: errors.append(v))

    from physion.gui.main import MainWindow
    window = MainWindow(qapp)
    window.slot_errors = errors
    yield window
    # delete the window now (not at interpreter exit -> segfault)
    window.close()
    window.deleteLater()
    qapp.processEvents()
