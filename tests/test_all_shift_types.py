"""
Smoke tests for all shift types: construct, compute scores, and plot.

These tests verify that the full pipeline runs without errors for each
shift class, using both simple and more realistic inputs.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from shifterator import (
    WeightedAvgShift,
    ProportionShift,
    EntropyShift,
    KLDivergenceShift,
    JSDivergenceShift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_freqs():
    """Two simple frequency distributions with some overlap."""
    freq1 = {"happy": 20, "sad": 5, "the": 50, "good": 15, "bad": 10}
    freq2 = {"happy": 10, "sad": 15, "the": 45, "good": 5, "bad": 20, "angry": 8}
    return freq1, freq2


@pytest.fixture
def identical_freqs():
    """Identical distributions — edge case for many metrics."""
    freq = {"a": 10, "b": 20, "c": 30}
    return freq.copy(), freq.copy()


@pytest.fixture
def simple_scores():
    """Simple sentiment scores."""
    return {
        "happy": 8.0,
        "sad": 2.0,
        "the": 5.0,
        "good": 7.0,
        "bad": 3.0,
        "angry": 2.5,
    }


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# WeightedAvgShift
# ---------------------------------------------------------------------------

class TestWeightedAvgShift:
    def test_with_explicit_scores(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_with_lexicon(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1="labMT_English", handle_missing_scores="exclude")
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_plot(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        ax = shift.get_shift_graph(top_n=5, show_plot=False)
        assert ax is not None

    def test_reference_value_average(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(
            freq1, freq2,
            type2score_1=simple_scores, type2score_2=simple_scores,
            reference_value="average",
        )
        assert shift.reference_value == shift.get_weighted_score(shift.type2freq_1, shift.type2score_1)

    def test_stop_words(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(
            freq1, freq2,
            type2score_1=simple_scores, type2score_2=simple_scores,
            stop_words={"the"},
        )
        assert "the" not in shift.types

    def test_normalization_trajectory(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(
            freq1, freq2,
            type2score_1=simple_scores, type2score_2=simple_scores,
            normalization="trajectory",
        )
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_uniform_scores_default(self):
        """When no scores given, defaults to uniform (1) scores."""
        freq1 = {"a": 10, "b": 20}
        freq2 = {"a": 15, "b": 15}
        shift = WeightedAvgShift(freq1, freq2)
        for t in shift.types:
            assert shift.type2score_1[t] == 1
            assert shift.type2score_2[t] == 1


# ---------------------------------------------------------------------------
# ProportionShift
# ---------------------------------------------------------------------------

class TestProportionShift:
    def test_basic(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = ProportionShift(freq1, freq2)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_plot(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = ProportionShift(freq1, freq2)
        ax = shift.get_shift_graph(top_n=5, show_plot=False)
        assert ax is not None

    def test_identical_systems(self, identical_freqs):
        freq1, freq2 = identical_freqs
        shift = ProportionShift(freq1, freq2)
        scores = shift.get_shift_scores()
        for s in scores.values():
            assert s == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# EntropyShift
# ---------------------------------------------------------------------------

class TestEntropyShift:
    def test_shannon(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = EntropyShift(freq1, freq2, base=2, alpha=1)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_tsallis_alpha_2(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = EntropyShift(freq1, freq2, alpha=2)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_tsallis_alpha_half(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = EntropyShift(freq1, freq2, alpha=0.5)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_plot(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = EntropyShift(freq1, freq2)
        ax = shift.get_shift_graph(top_n=5, show_plot=False)
        assert ax is not None

    def test_identical_systems(self, identical_freqs):
        freq1, freq2 = identical_freqs
        shift = EntropyShift(freq1, freq2)
        assert shift.diff == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# KLDivergenceShift
# ---------------------------------------------------------------------------

class TestKLDivergenceShift:
    def test_basic(self):
        # KLD requires all types in freq2 to also be in freq1
        freq1 = {"a": 10, "b": 20, "c": 30}
        freq2 = {"a": 15, "b": 15, "c": 30}
        shift = KLDivergenceShift(freq1, freq2)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_plot(self):
        freq1 = {"a": 10, "b": 20, "c": 30}
        freq2 = {"a": 15, "b": 15, "c": 30}
        shift = KLDivergenceShift(freq1, freq2)
        ax = shift.get_shift_graph(top_n=3, show_plot=False)
        assert ax is not None

    def test_missing_type_raises(self):
        """KLD is undefined if freq2 has types not in freq1."""
        freq1 = {"a": 10, "b": 20}
        freq2 = {"a": 15, "c": 30}  # "c" not in freq1
        with pytest.raises(ValueError):
            KLDivergenceShift(freq1, freq2)

    def test_identical_systems(self, identical_freqs):
        freq1, freq2 = identical_freqs
        shift = KLDivergenceShift(freq1, freq2)
        # KL(P || P) = 0
        assert shift.diff == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# JSDivergenceShift
# ---------------------------------------------------------------------------

class TestJSDivergenceShift:
    def test_basic(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = JSDivergenceShift(freq1, freq2)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_plot(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = JSDivergenceShift(freq1, freq2)
        ax = shift.get_shift_graph(top_n=5, show_plot=False)
        assert ax is not None

    def test_identical_systems(self, identical_freqs):
        freq1, freq2 = identical_freqs
        shift = JSDivergenceShift(freq1, freq2)
        assert shift.diff == pytest.approx(0.0, abs=1e-10)

    def test_bad_weights_raises(self, simple_freqs):
        freq1, freq2 = simple_freqs
        with pytest.raises(ValueError):
            JSDivergenceShift(freq1, freq2, weight_1=0.3, weight_2=0.3)

    def test_custom_weights(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = JSDivergenceShift(freq1, freq2, weight_1=0.3, weight_2=0.7)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)

    def test_tsallis_alpha(self, simple_freqs):
        freq1, freq2 = simple_freqs
        shift = JSDivergenceShift(freq1, freq2, alpha=2)
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)


# ---------------------------------------------------------------------------
# Shift base class methods (via WeightedAvgShift)
# ---------------------------------------------------------------------------

class TestShiftBaseMethods:
    def test_get_shift_scores_details(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        result = shift.get_shift_scores(details=True)
        p_diff, s_diff, p_avg, s_ref_diff, shift_scores = result
        assert isinstance(p_diff, dict)
        assert isinstance(s_diff, dict)
        assert isinstance(p_avg, dict)
        assert isinstance(s_ref_diff, dict)
        assert isinstance(shift_scores, dict)

    def test_get_shift_component_sums(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        comp_sums = shift.get_shift_component_sums()
        expected_keys = {"pos_s_pos_p", "pos_s_neg_p", "neg_s_pos_p", "neg_s_neg_p", "pos_s", "neg_s"}
        assert set(comp_sums.keys()) == expected_keys

    def test_get_weighted_score(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        s_avg = shift.get_weighted_score(shift.type2freq_1, shift.type2score_1)
        assert isinstance(s_avg, float)
        assert s_avg > 0

    def test_plot_with_filename(self, simple_freqs, simple_scores, tmp_path):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        filepath = tmp_path / "test_shift.png"
        shift.get_shift_graph(top_n=5, show_plot=False, filename=str(filepath))
        assert filepath.exists()

    def test_plot_not_detailed(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        ax = shift.get_shift_graph(top_n=5, show_plot=False, detailed=False)
        assert ax is not None

    def test_plot_no_insets(self, simple_freqs, simple_scores):
        freq1, freq2 = simple_freqs
        shift = WeightedAvgShift(freq1, freq2, type2score_1=simple_scores, type2score_2=simple_scores)
        ax = shift.get_shift_graph(
            top_n=5, show_plot=False,
            text_size_inset=False, cumulative_inset=False,
        )
        assert ax is not None
