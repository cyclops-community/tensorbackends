"""
TensorBackends
"""

from .version import VERSION as __version__

from . import interface
from .backends import get, isavailable
from . import utils
