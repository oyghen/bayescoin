__all__ = ("__version__", "BetaShape", "hdi")

from importlib import metadata

from bayescoin.core import BetaShape, hdi

__version__ = metadata.version(__name__)
