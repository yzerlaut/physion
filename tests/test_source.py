r"""
every source file compiles, without SyntaxWarning
    (e.g. invalid escape sequences like '$\Delta$' -> use '$\\Delta$',
     which will become errors in future python versions)
"""
import pathlib, warnings
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'physion'
FILES = sorted(p for p in SRC.rglob('*.py') if 'plot_tools' not in p.parts)


@pytest.mark.parametrize('path', FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_compiles_without_warning(path):
    with warnings.catch_warnings():
        warnings.simplefilter('error', SyntaxWarning)
        compile(path.read_bytes(), str(path), 'exec')
