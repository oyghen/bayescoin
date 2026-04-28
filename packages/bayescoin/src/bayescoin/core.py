__all__ = ["BetaShape"]

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from scipy import optimize, stats


@dataclass(frozen=True, slots=True)
class BetaShape:
    """Shape parameters of the Beta distribution."""

    a: int | float
    b: int | float

    def __init__(self, a: int | float, b: int | float) -> None:
        # normalize to floats for internal consistency and cache stability
        a = float(a)
        b = float(b)

        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("a and b must be finite numbers")
        if a <= 0.0 or b <= 0.0:
            raise ValueError("a and b must be > 0")

        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    def __repr__(self) -> str:
        # show parameters as integers when they are, otherwise as float
        def fmt(x: float) -> str:
            return str(int(x)) if x.is_integer() else repr(x)

        return f"BetaShape(a={fmt(self.a)}, b={fmt(self.b)})"

    # --- summaries
    @property
    def mean(self) -> float:
        """Return the mean of the Beta distribution."""
        return self.a / (self.a + self.b)

    @property
    def mode(self) -> float | None:
        """Return the mode of the Beta distribution or None when undefined."""
        a, b = self.a, self.b
        if a > 1.0 and b > 1.0:
            return (a - 1.0) / (a + b - 2.0)
        return None

    def hdi(self, level: float = 0.95) -> tuple[float, float] | None:
        """Return the highest density interval (HDI) at specified credibility level."""
        a, b = self.a, self.b
        if a <= 1.0 or b <= 1.0:
            return None

        p = round(float(level), 12)
        if not (math.isfinite(p) and 0.0 < p < 1.0):
            raise ValueError("credibility level must be a finite number in (0, 1)")

        return _hdi_cached(a, b, p)

    # --- posterior updating
    def posterior_from_observations(
        self,
        data: Iterable[int | str],
        success_value: int | str = 1,
    ) -> "BetaShape":
        """Return posterior Beta parameters updated from raw observations."""
        successes = 0
        trials = 0
        for observation in data:
            trials += 1
            if observation == success_value:
                successes += 1
        return self.posterior_from_counts(successes, trials)

    def posterior_from_counts(self, successes: int, trials: int) -> "BetaShape":
        """Return posterior Beta parameters updated from aggregated counts."""
        if not (isinstance(successes, int) and isinstance(trials, int)):
            raise TypeError("successes and trials must be integers")
        if successes < 0 or trials < 0:
            raise ValueError("successes and trials must be >= 0")
        if successes > trials:
            raise ValueError("successes cannot exceed trials")

        failures = trials - successes

        return BetaShape(a=self.a + successes, b=self.b + failures)

    # --- convenience
    @classmethod
    def uniform(cls) -> "BetaShape":
        """Return parameters of a weakly informative uniform prior Beta(1, 1)."""
        return cls(a=1, b=1)

    @classmethod
    def jeffreys(cls) -> "BetaShape":
        """Return parameters of a non-informative Jeffreys prior Beta(0.5, 0.5)."""
        return cls(a=0.5, b=0.5)

    def to_dist(self):
        """Return a SciPy Beta distribution parameterized by a and b."""
        return stats.beta(self.a, self.b)

    def summary(self, hdi_level: float = 0.95) -> str:
        """Return a concise one-line summary."""
        parts = [f"{self!r} ->", f"mean={self.mean:g}"]

        mode = self.mode
        if mode is not None:
            parts.append(f"{mode=:g}")

        hdi = self.hdi(hdi_level)
        if hdi is not None:
            lower, upper = hdi
            parts.append(f"{100 * hdi_level:g}%-HDI=[{lower:g}, {upper:g}]")

        return " ".join(parts)


@lru_cache(maxsize=256)
def _hdi_cached(a: float, b: float, p: float) -> tuple[float, float]:
    """Return the HDI for a Beta(a, b) at specified credibility level p (cached)."""
    dist = stats.beta(a, b)
    tail = 1.0 - p

    def width(x: float) -> float:
        return dist.ppf(p + x) - dist.ppf(x)

    res = optimize.minimize_scalar(
        width,
        bounds=(0.0, tail),
        method="bounded",
        options={"xatol": 1e-12},
    )

    if not getattr(res, "success", True):
        msg = getattr(res, "message", "minimization did not succeed")
        raise RuntimeError(f"optimization failed while computing HDI: {msg!s}")

    x_opt = float(res.x)
    lower, upper = dist.ppf([x_opt, x_opt + p])
    return float(lower), float(upper)
