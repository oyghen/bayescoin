__all__ = ("__version__", "BetaShape", "plot")

from importlib import metadata

from bayescoin.core import BetaShape
from bayescoin.viz import plot

__version__ = metadata.version(__name__)
