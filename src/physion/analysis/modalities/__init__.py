"""
the modalities of the NWB files, read by physion.analysis.read_NWB.Data

one module per modality, each one defining the methods (has_X, build_X, ...)
of a "mixin" class inherited by the Data class
"""
from .visual_stim import VisualStimMixin
from .locomotion import LocomotionMixin
from .facecamera import FaceCameraMixin
from .opto import OptoMixin
from .ophys import OphysMixin
from .ephys import EphysMixin
