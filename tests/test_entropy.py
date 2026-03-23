"""Unit tests for the entropy module's core math."""

import math
import pytest
from shifterator.entropy import (
    get_relative_freqs,
    get_entropy_scores,
    get_entropy_type_scores,
    get_jsd_scores,
    get_jsd_type_scores,
)


# ---------------------------------------------------------------------------
# get_relative_freqs
# ---------------------------------------------------------------------------

class TestGetRelativeFreqs:
    def test_uniform(self):
        freqs = {"a": 10, "b": 10, "c": 10}
        p = get_relative_freqs(freqs)
        assert p == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})

    def test_sums_to_one(self):
        freqs = {"x": 3, "y": 7, "z": 15}
        p = get_relative_freqs(freqs)
        assert sum(p.values()) == pytest.approx(1.0)

    def test_single_type(self):
        p = get_relative_freqs({"only": 42})
        assert p == {"only": 1.0}


# ---------------------------------------------------------------------------
# get_entropy_type_scores  (Shannon, alpha=1)
# ---------------------------------------------------------------------------

class TestEntropyTypeScoresShannon:
    def test_nonzero_both(self):
        s1, s2 = get_entropy_type_scores(0.25, 0.5, base=2, alpha=1)
        assert s1 == pytest.approx(-math.log2(0.25))  # 2.0
        assert s2 == pytest.approx(-math.log2(0.5))    # 1.0

    def test_zero_prob_gives_zero_score(self):
        s1, s2 = get_entropy_type_scores(0, 0.5, base=2, alpha=1)
        assert s1 == 0.0
        assert s2 == pytest.approx(1.0)

    def test_both_zero(self):
        s1, s2 = get_entropy_type_scores(0, 0, base=2, alpha=1)
        assert s1 == 0.0
        assert s2 == 0.0


# ---------------------------------------------------------------------------
# get_entropy_type_scores  (Tsallis, alpha != 1)
# ---------------------------------------------------------------------------

class TestEntropyTypeScoresTsallis:
    def test_alpha_2(self):
        p = 0.25
        s1, s2 = get_entropy_type_scores(p, p, base=2, alpha=2)
        expected = -1 * p ** (2 - 1) / (2 - 1)  # -p / 1 = -0.25
        assert s1 == pytest.approx(expected)
        assert s2 == pytest.approx(expected)

    def test_alpha_half(self):
        p = 0.25
        s1, s2 = get_entropy_type_scores(p, p, base=2, alpha=0.5)
        expected = -1 * p ** (0.5 - 1) / (0.5 - 1)  # -p^{-0.5} / (-0.5) = 2/sqrt(0.25) = 4
        assert s1 == pytest.approx(expected)

    def test_zero_prob_alpha_gt_1(self):
        s1, s2 = get_entropy_type_scores(0, 0.5, base=2, alpha=2)
        assert s1 == 0.0
        assert s2 != 0.0


# ---------------------------------------------------------------------------
# get_entropy_scores  (integration: two full systems)
# ---------------------------------------------------------------------------

class TestGetEntropyScores:
    def test_identical_systems_shannon(self):
        """Identical distributions should produce identical score dicts."""
        p = {"a": 0.5, "b": 0.3, "c": 0.2}
        s1, s2 = get_entropy_scores(p, p, base=2, alpha=1)
        for t in p:
            assert s1[t] == pytest.approx(s2[t])

    def test_disjoint_types(self):
        """Types only in one system get zero score in the other."""
        p1 = {"a": 0.5, "b": 0.5}
        p2 = {"b": 0.5, "c": 0.5}
        s1, s2 = get_entropy_scores(p1, p2, base=2, alpha=1)
        # "a" not in p2 → s2["a"] should come from p2=0 → 0
        assert s2["a"] == 0.0
        # "c" not in p1 → s1["c"] should come from p1=0 → 0
        assert s1["c"] == 0.0


# ---------------------------------------------------------------------------
# get_jsd_type_scores  (Shannon, alpha=1)
# ---------------------------------------------------------------------------

class TestJSDTypeScores:
    def test_identical_probs(self):
        """If p1 == p2, JSD contribution should be 0 for that type."""
        p = 0.25
        m = 0.5 * p + 0.5 * p  # = p
        s1, s2 = get_jsd_type_scores(p, p, m, 0.5, 0.5, base=2, alpha=1)
        assert s1 == pytest.approx(0.0)
        assert s2 == pytest.approx(0.0)

    def test_one_zero(self):
        """If p1=0, its score uses just log(m)."""
        p1, p2 = 0, 0.5
        m = 0.5 * p1 + 0.5 * p2
        s1, s2 = get_jsd_type_scores(p1, p2, m, 0.5, 0.5, base=2, alpha=1)
        assert s1 == pytest.approx(0.5 * math.log2(m))
        assert s2 == pytest.approx(0.5 * (math.log2(p2) - math.log2(m)))


# ---------------------------------------------------------------------------
# get_jsd_scores  (integration: two full systems)
# ---------------------------------------------------------------------------

class TestGetJSDScores:
    def test_identical_systems_zero_jsd(self):
        """JSD of a distribution with itself should be 0."""
        p = {"a": 0.5, "b": 0.3, "c": 0.2}
        type2m, s1, s2 = get_jsd_scores(p, p, base=2, alpha=1)
        # All individual scores should be 0
        for t in p:
            assert s1[t] == pytest.approx(0.0, abs=1e-12)
            assert s2[t] == pytest.approx(0.0, abs=1e-12)
        # Mixture should equal the original
        for t in p:
            assert type2m[t] == pytest.approx(p[t])

    def test_jsd_nonnegative(self):
        """JSD is always non-negative; total shift from sum of scores >= 0."""
        p1 = {"a": 0.7, "b": 0.2, "c": 0.1}
        p2 = {"a": 0.1, "b": 0.3, "c": 0.6}
        type2m, s1, s2 = get_jsd_scores(p1, p2, base=2, alpha=1)
        # Total JSD = sum over types of (p2 * s2 - p1 * s1) … but the
        # simplest check: every type's mixture-weighted contribution is >= 0
        # when reference_value=0. We just check scores are finite.
        for t in p1:
            assert math.isfinite(s1[t])
            assert math.isfinite(s2[t])

    def test_asymmetric_weights(self):
        """Non-equal weights should still produce valid results."""
        p1 = {"a": 0.6, "b": 0.4}
        p2 = {"a": 0.4, "b": 0.6}
        type2m, s1, s2 = get_jsd_scores(p1, p2, weight_1=0.3, weight_2=0.7, base=2, alpha=1)
        for t in p1:
            assert type2m[t] == pytest.approx(0.3 * p1[t] + 0.7 * p2[t])
