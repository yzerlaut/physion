"""
analysis on a real NWB file (skipped when the file is not available)

    set the PHYSION_TEST_NWB environment variable to point to another file
"""
import os
import numpy as np
import pytest

NWB = os.environ.get('PHYSION_TEST_NWB',
        os.path.expanduser('~/DATA/physion_Demo-Datasets/PYR-WT/2025_11_14-13-54-32.nwb'))

pytestmark = pytest.mark.skipif(not os.path.isfile(NWB),
                                reason='no test NWB file at "%s"' % NWB)


@pytest.fixture(scope='module')
def data():
    from physion.analysis.read_NWB import Data
    d = Data(NWB, verbose=False)
    yield d
    d.close()


def test_read_NWB(data):
    assert len(data.protocols)>0
    assert data.tlim[1]>data.tlim[0]
    if data.has_ophys():
        data.build_dFoF(verbose=False)
        assert data.dFoF.shape[1] == len(data.t_dFoF)


def test_EpisodeData(data):
    from physion.analysis.episodes.build import EpisodeData
    quantity = 'dFoF' if data.has_ophys() else 'photodiode'
    ep = EpisodeData(data, protocol_id=0, quantities=[quantity])
    response = getattr(ep, quantity)
    assert response.shape[0] == len(ep.time_start) # one row per episode
    assert response.shape[-1] == len(ep.t)
    assert np.isfinite(response).all()
