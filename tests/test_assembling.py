"""
NWB assembling
"""
import types
import pytest

from physion.assembling.add_ophys import add_ophys


def test_add_ophys_without_xml_file(tmp_path):
    """ a clear error when the imaging folder has no Bruker xml file """
    args = types.SimpleNamespace(imaging=str(tmp_path))
    with pytest.raises(FileNotFoundError, match='no xml file'):
        add_ophys(None, args, metadata={})
