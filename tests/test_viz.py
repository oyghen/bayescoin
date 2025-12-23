import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection

import bayescoin

matplotlib.use("Agg")


class TestPlot:
    def test_unsupported_first_argument_raises(self):
        with pytest.raises(TypeError):
            bayescoin.plot("string input not supported")

    @pytest.mark.parametrize(
        "a, b, hdi_level",
        [
            (2.0, 2.0, 0.5),
            (3.0, 6.0, 0.9),
            (8.0, 4.0, 0.8),
        ],
    )
    def test_numeric_input(self, a: float, b: float, hdi_level: float):
        ax = bayescoin.plot(a, b, hdi_level=hdi_level, num_points=512)
        assert isinstance(ax, Axes)
        assert ax.lines, "expected at least one plotted line"

        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert polys, "HDI region was not shaded"

        besh = bayescoin.BetaShape(2, 2)
        lower, upper = besh.hdi(hdi_level)
        xdata = ax.lines[0].get_xdata()

        assert lower >= xdata.min()
        assert upper <= xdata.max()

    @pytest.mark.parametrize(
        "besh, hdi_level",
        [
            (bayescoin.BetaShape(2, 2), 0.5),
            (bayescoin.BetaShape(3, 6), 0.9),
            (bayescoin.BetaShape(8, 4), 0.8),
        ],
    )
    def test_betashape_input(self, besh: bayescoin.BetaShape, hdi_level: float):
        ax = bayescoin.plot(besh, hdi_level=hdi_level, num_points=512)
        assert isinstance(ax, Axes)
        assert ax.lines, "expected at least one plotted line"

        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert polys, "HDI region was not shaded"

        lower, upper = besh.hdi(hdi_level)
        xdata = ax.lines[0].get_xdata()

        assert lower >= xdata.min()
        assert upper <= xdata.max()

    def test_axes_passthrough(self):
        _, ax = plt.subplots()
        result = bayescoin.plot(1.0, 1.0, ax=ax)
        assert result is ax

        _, ax = plt.subplots()
        result = bayescoin.plot(bayescoin.BetaShape.uniform(), ax=ax)
        assert result is ax
