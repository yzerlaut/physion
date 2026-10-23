"""
entry point of the stimulus presentation process (see show.py)

run with runpy.run_path, with "cmd_queue", "status_queue", "progress" and
"config" given as globals. The presenter and movie modules are loaded from
their files, WITHOUT importing the physion package (whose __init__ imports
all the acquisition, hardware and analysis modules): fast start, nothing
else loaded in the presentation process.
"""
import os, sys, importlib.util

status_queue.put(('stage', 'process started')) # noqa: F821 (globals from runpy)

_here = os.path.dirname(os.path.abspath(__file__))

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_movie = _load('_physion_stim_movie', 'movie.py')
_presenter = _load('_physion_stim_presenter', 'presenter.py')
_presenter.movie_module = _movie

_presenter.run_presenter(cmd_queue, status_queue, progress, config) # noqa: F821
