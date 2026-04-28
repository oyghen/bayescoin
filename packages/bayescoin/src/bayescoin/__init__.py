__all__ = ("__version__", "BetaShape", "plot")

from importlib import metadata

from bayescoin.core import BetaShape
from bayescoin.plot import plot

__version__ = metadata.version(__name__)
