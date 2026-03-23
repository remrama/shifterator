"""Tests for the helper module: preprocessing and lexicon loading."""

import pytest
from shifterator.helper import preprocess_words_scores, get_score_dictionary


# ---------------------------------------------------------------------------
# get_score_dictionary
# ---------------------------------------------------------------------------

class TestGetScoreDictionary:
    def test_dict_passthrough(self):
        """Passing a dict should return a copy of it with None ref."""
        d = {"happy": 8.0, "sad": 2.0}
        scores, ref = get_score_dictionary(d)
        assert scores == d
        assert ref is None
        # Should be a copy, not the same object
        assert scores is not d

    def test_labmt_english(self):
        """Loading the labMT English lexicon should give a dict with known words."""
        scores, ref = get_score_dictionary("labMT_English")
        assert isinstance(scores, dict)
        assert len(scores) > 100
        assert "happiness" in scores or "happy" in scores
        assert ref == 5  # labMT reference is 5

    def test_invalid_lexicon_raises(self):
        with pytest.raises(Exception):
            get_score_dictionary("nonexistent_lexicon_xyz")


# ---------------------------------------------------------------------------
# preprocess_words_scores
# ---------------------------------------------------------------------------

class TestPreprocessWordsScores:
    def test_basic_passthrough(self):
        """Words present in both freq and score dicts pass through."""
        freq1 = {"a": 10, "b": 5}
        freq2 = {"a": 7, "b": 8}
        score1 = {"a": 1.0, "b": 2.0}
        score2 = {"a": 1.5, "b": 2.5}
        result = preprocess_words_scores(freq1, score1, freq2, score2, [], set(), "error")
        f1, f2, s1, s2, types, filtered, no_score, adopted = result
        assert types == {"a", "b"}
        assert len(filtered) == 0
        assert len(no_score) == 0

    def test_stop_words_excluded(self):
        freq1 = {"a": 10, "b": 5, "the": 100}
        freq2 = {"a": 7, "b": 8, "the": 200}
        score1 = {"a": 1.0, "b": 2.0, "the": 3.0}
        score2 = {"a": 1.5, "b": 2.5, "the": 3.5}
        result = preprocess_words_scores(freq1, score1, freq2, score2, [], {"the"}, "error")
        _, _, _, _, types, filtered, _, _ = result
        assert "the" not in types
        assert "the" in filtered

    def test_missing_score_error(self):
        freq1 = {"a": 10, "b": 5}
        freq2 = {"a": 7, "b": 8}
        score1 = {"a": 1.0}  # "b" missing
        score2 = {"a": 1.5, "b": 2.5}
        with pytest.raises(KeyError):
            preprocess_words_scores(freq1, score1, freq2, score2, [], set(), "error")

    def test_missing_score_exclude(self):
        freq1 = {"a": 10, "b": 5}
        freq2 = {"a": 7, "b": 8}
        score1 = {"a": 1.0}
        score2 = {"a": 1.5, "b": 2.5}
        result = preprocess_words_scores(freq1, score1, freq2, score2, [], set(), "exclude")
        _, _, _, _, types, _, no_score, _ = result
        assert "b" not in types
        assert "b" in no_score

    def test_missing_score_adopt(self):
        freq1 = {"a": 10, "b": 5}
        freq2 = {"a": 7, "b": 8}
        score1 = {"a": 1.0}
        score2 = {"a": 1.5, "b": 2.5}
        result = preprocess_words_scores(freq1, score1, freq2, score2, [], set(), "adopt")
        _, _, s1, _, types, _, _, adopted = result
        assert "b" in types
        assert "b" in adopted
        assert s1["b"] == 2.5  # adopted from score2

    def test_stop_lens(self):
        """Words with scores in stop_lens range should be filtered."""
        freq1 = {"a": 10, "b": 5}
        freq2 = {"a": 7, "b": 8}
        score1 = {"a": 1.0, "b": 5.0}
        score2 = {"a": 1.5, "b": 5.0}
        # Filter scores between 4.5 and 5.5
        result = preprocess_words_scores(freq1, score1, freq2, score2, [(4.5, 5.5)], set(), "error")
        _, _, _, _, types, filtered, _, _ = result
        assert "b" not in types
        assert "b" in filtered

    def test_zero_freq_for_missing_types(self):
        """Types in one system but not the other get freq=0."""
        freq1 = {"a": 10}
        freq2 = {"a": 7, "b": 8}
        score1 = {"a": 1.0, "b": 2.0}
        score2 = {"a": 1.5, "b": 2.5}
        result = preprocess_words_scores(freq1, score1, freq2, score2, [], set(), "error")
        f1, f2, _, _, types, _, _, _ = result
        assert "b" in types
        assert f1["b"] == 0
