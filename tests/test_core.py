import math
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from typing import TypeAlias

import pytest
from scipy import stats

from bayescoin import BetaShape

ContextManager: TypeAlias = (
    nullcontext[None] | pytest.RaisesExc[TypeError] | pytest.RaisesExc[ValueError]
)


class TestBetaShapeInitAndRepresentation:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (1, 2, "BetaShape(a=1, b=2)"),
            (1.0, 3.0, "BetaShape(a=1, b=3)"),
            (0.5, 0.5, "BetaShape(a=0.5, b=0.5)"),
        ],
    )
    def test_repr(self, a: int | float, b: int | float, expected: str):
        result = BetaShape(a, b)
        assert repr(result) == expected

    def test_parameters_are_floats(self):
        result = BetaShape(1, 1)
        assert isinstance(result.a, float)
        assert isinstance(result.b, float)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (1, 1),
            (0.5, 0.5),
            (2.3, 4.7),
        ],
    )
    def test_frozen_dataclass(self, a: int | float, b: int | float):
        result = BetaShape(a, b)
        with pytest.raises(FrozenInstanceError):
            result.a = 999

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (0, 0),
            (0, 1),
            (1, 0),
            (-1, -1),
            (-1, 2),
            (1, -2),
            (math.inf, math.inf),
            (math.inf, 1),
            (1, math.inf),
            (math.nan, math.nan),
            (math.nan, 1),
            (1, math.nan),
        ],
        ids=[
            "zero values",
            "zero a",
            "zero b",
            "negative values",
            "negative a",
            "negative b",
            "non-finite values",
            "non-finite a",
            "non-finite b",
            "NaN values",
            "NaN a",
            "NaN b",
        ],
    )
    def test_invalid_init(self, a: int | float, b: int | float):
        with pytest.raises(ValueError):
            BetaShape(a, b)


class TestBetaShapeSummaries:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (1, 1, 0.5),
            (2, 2, 0.5),
            (2, 3, 2.0 / 5.0),
            (5, 2, 5 / 7),
            (0.5, 0.5, 0.5),
        ],
    )
    def test_mean(self, a: int | float, b: int | float, expected: float):
        result = BetaShape(a, b)
        dist = stats.beta(a, b)
        assert result.mean == pytest.approx(expected)
        assert result.mean == pytest.approx(dist.mean())

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            # defined when both > 1
            (2, 3, (2 - 1) / (2 + 3 - 2)),
            (3, 3, 0.5),
            # undefined when <=1
            (1, 1, None),
            (0.5, 2, None),
            (2, 0.8, None),
        ],
    )
    def test_mode(self, a: int | float, b: int | float, expected: float | None):
        result = BetaShape(a, b)
        assert result.mode == expected

    @pytest.mark.parametrize(
        ("a", "b", "hdi_level", "expected"),
        [
            (2, 2, 0.95, (0.094299, 0.905701)),
            (2, 2, 0.90, (0.135350, 0.864649)),
            (4, 4, 0.95, (0.184051, 0.815948)),
            (4, 4, 0.90, (0.225321, 0.774678)),
            (100, 100, 0.95, (0.430950, 0.569049)),
            (100, 100, 0.90, (0.441969, 0.558030)),
            (3, 5, 0.85, (0.128154, 0.595574)),
            (12, 8, 0.8, (0.466301, 0.743738)),
        ],
    )
    def test_hdi(
        self,
        a: int | float,
        b: int | float,
        hdi_level: float,
        expected: tuple[float, float],
    ):
        result = BetaShape(a, b)
        lower, upper = result.hdi(hdi_level)

        dist = stats.beta(result.a, result.b)
        prob = dist.cdf(upper) - dist.cdf(lower)

        assert result.mean == pytest.approx(dist.mean(), abs=1e-6)
        assert 0.0 <= lower < result.mean < upper <= 1.0
        assert lower == pytest.approx(expected[0], abs=1e-6)
        assert upper == pytest.approx(expected[1], abs=1e-6)
        assert prob == pytest.approx(hdi_level, rel=1e-6, abs=1e-6)

    def test_invalid_parameters_returns_none(self):
        assert BetaShape(1, 1).hdi(0.95) is None

    @pytest.mark.parametrize("bad_value", [0.0, 1.0, 1.1, -0.1, math.nan, math.inf])
    def test_invalid_credibility_level_raises(self, bad_value: float):
        with pytest.raises(ValueError):
            BetaShape(2, 2).hdi(bad_value)

    def test_hdi_is_cached(self):
        result = BetaShape(2, 5)
        level = 0.95

        # repeated calls should return identical results (and benefit from caching)
        first = result.hdi(level)
        second = result.hdi(level)

        assert first == second


class TestPosteriorUpdatingFromObservations:
    def test_coin_tosses(self):
        prior = BetaShape(a=1, b=1)
        data = ["H", "H", "T", "H", "T", "H", "H", "H", "T", "T"]  # 6 / 10 successes
        successes = sum(obs == "H" for obs in data)
        failures = sum(obs != "H" for obs in data)

        post = prior.posterior_from_observations(data, success_value="H")
        lower, upper = post.hdi(0.95)

        assert post.a == prior.a + successes
        assert post.b == prior.b + failures
        assert post.mean == pytest.approx(7 / 12)
        assert lower == pytest.approx(0.318, abs=1e-3)
        assert upper == pytest.approx(0.841, abs=1e-3)

    def test_posterior_from_observations_with_ints(self):
        prior = BetaShape(a=2, b=3)
        data = [1, 0, 1, 1, 0]  # 3 successes, 5 trials
        post = prior.posterior_from_observations(data)
        assert post.a == pytest.approx(2 + 3)
        assert post.b == pytest.approx(3 + (5 - 3))

    def test_posterior_from_observations_with_strings_and_custom_success_value(self):
        prior = BetaShape(a=1, b=1)
        data = ["yes", "no", "yes", "no", "maybe"]  # 2 successes, 5 trials
        post = prior.posterior_from_observations(data, success_value="yes")
        assert post.a == pytest.approx(1 + 2)
        assert post.b == pytest.approx(1 + (5 - 2))

    def test_posterior_from_observations_with_iterator(self):
        prior = BetaShape(a=5, b=4)
        data = (i for i in [1, 1, 0])  # 2 successes, 3 trials
        post = prior.posterior_from_observations(data)
        assert post.a == pytest.approx(5 + 2)
        assert post.b == pytest.approx(4 + (3 - 2))

    @pytest.mark.parametrize(
        ("data", "prior", "expected"),
        [
            ([1, 0, 1, 1, 0], BetaShape(a=1, b=1), BetaShape(a=4, b=3)),
            ([1, 0, 1, 1, 0], BetaShape(a=2, b=2), BetaShape(a=5, b=4)),
        ],
    )
    def test_posterior_update(
        self,
        data: list[int],
        prior: BetaShape,
        expected: BetaShape,
    ):
        post = prior.posterior_from_observations(data)
        assert isinstance(post, BetaShape)
        assert post == expected

        for prop in ["a", "b", "mean", "mode"]:
            actual_value = getattr(post, prop)
            expected_value = getattr(expected, prop)
            assert actual_value == expected_value


class TestPosteriorUpdatingFromCount:
    @pytest.mark.parametrize(
        ("successes", "trials"),
        [
            (5, 10),
            (4, 10),
            (6, 10),
            (0, 10),
            (10, 10),
        ],
        ids=[
            "even successes",
            "less successes",
            "more successes",
            "zero successes",
            "only successes",
        ],
    )
    def test_posterior_from_counts(self, successes: int, trials: int):
        prior = BetaShape(a=1, b=1)
        failures = trials - successes

        post = prior.posterior_from_counts(successes, trials)

        assert isinstance(post, BetaShape)
        assert post.a == pytest.approx(prior.a + successes)
        assert post.b == pytest.approx(prior.b + failures)

    def test_zero_trials_count_update(self):
        prior = BetaShape(a=1, b=1)

        # 0 trials → no new information, posterior equals prior
        post = prior.posterior_from_counts(successes=0, trials=0)

        assert post is not prior
        assert post == prior

        for prop in ["a", "b", "mean", "mode"]:
            actual_value = getattr(post, prop)
            expected_value = getattr(prior, prop)
            assert actual_value == expected_value

    @pytest.mark.parametrize(
        ("successes", "trials", "ctx"),
        [
            (1.0, 2.0, pytest.raises(TypeError)),
            (1.0, 2, pytest.raises(TypeError)),
            (1, 2.0, pytest.raises(TypeError)),
            (-1, -2, pytest.raises(ValueError)),
            (-1, 2, pytest.raises(ValueError)),
            (1, -2, pytest.raises(ValueError)),
            (2, 1, pytest.raises(ValueError)),
        ],
        ids=[
            "float values",
            "float successes",
            "float trials",
            "negative values",
            "negative successes",
            "negative trials",
            "successes > trials",
        ],
    )
    def test_posterior_from_counts_type_and_value_checks(
        self,
        successes: int,
        trials: int,
        ctx: ContextManager,
    ):
        prior = BetaShape(a=1, b=1)
        with ctx:
            prior.posterior_from_counts(successes, trials)


class TestConveniencePriors:
    def test_uniform(self):
        result = BetaShape.uniform()
        assert isinstance(result, BetaShape)
        assert result.a == pytest.approx(1.0)
        assert result.b == pytest.approx(1.0)
        assert result.mean == pytest.approx(0.5)
        assert result.mode is None

    def test_jeffreys(self):
        result = BetaShape.jeffreys()
        assert isinstance(result, BetaShape)
        assert result.a == pytest.approx(0.5)
        assert result.b == pytest.approx(0.5)
        assert result.mean == pytest.approx(0.5)
        assert result.mode is None


class TestScipyObject:
    def test_to_dist_matches_parameters(self):
        beta_shape = BetaShape(a=2.5, b=4.5)
        result = beta_shape.to_dist()
        quantile = result.ppf(0.25)
        assert result.mean() == pytest.approx(beta_shape.mean)
        assert 0.0 <= quantile <= 1.0


class TestSummaryString:
    def test_summary_contains_expected_parts(self):
        result = BetaShape(a=2, b=2).summary(hdi_level=0.9)
        assert "BetaShape" in result
        assert "mean=" in result
        assert "HDI" in result
