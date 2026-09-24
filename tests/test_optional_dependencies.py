"""
the optional dependencies (pip install "physion[ephys]") are not needed
    to launch the GUI or to build NWB files without ephys data
"""
import os, sys, subprocess, pathlib
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'

EPHYS = ['spikeinterface', 'probeinterface', 'elephant', 'neo', 'open_ephys']

SCRIPT = """
import sys, importlib
class Blocker: # simulates an install without the "ephys" extra
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in %r:
            raise ImportError('not installed: '+name)
sys.meta_path.insert(0, Blocker())
importlib.import_module(%r)
"""


@pytest.mark.parametrize('module', ['physion.gui.main',
                                    'physion.assembling.nwb',
                                    'physion.analysis.read_NWB'])
def test_import_without_ephys_extra(module):
    # in a fresh interpreter: the packages might be installed in this one
    env = dict(os.environ, PYTHONPATH=str(SRC), QT_QPA_PLATFORM='offscreen')
    result = subprocess.run([sys.executable, '-c', SCRIPT % (EPHYS, module)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
