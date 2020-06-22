"""
shifterator.py

TODO:
- Add funcs to shift class that allow for easy updating of type2freq dicts
- Make it easy to remove / reset the filter. This will involve having to hold
  onto stop words, their freqs, and their values (discarded as of now)
- Make it so you can specify words as stop words instead of just a filter window
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from .helper import *
from .plotting import *

# ------------------------------------------------------------------------------
# ---------------------------- GENERAL SHIFT CLASS -----------------------------
# ------------------------------------------------------------------------------
class Shift:
    """
    Shift object for calculating weighted scores of two systems of types,
    and the shift between them

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        Keys are types of a system and values are frequencies of those types
    type2score_1, type2score_2: dict or str, optional
        If dict, types are keys and values are scores associated with each
        type. If str, the name of a score lexicon included in Shifterator.
        If None and other type2score is None, defaults to uniform scores
        across types. Otherwise defaults to the other type2score dict
    reference_value: str or float, optional
        The reference score to use to partition scores into two different
        regimes. If 'average', uses the average score according to type2freq_1
        and type2score_1. If None and a lexicon is selected for type2score,
        uses the respective middle point in that lexicon's scale. Otherwise
        if None, uses zero as the reference point
    stop_lens: iterable of 2-tuples, optional
        Denotes intervals of scores that should be excluded when calculating
        shift scores, and types with scores in this range will be excluded
        from shift calculations
    normalization: str, optional
        If 'variation', normalizes shift scores so that the sum of
        their absolute values sums to 1. If 'trajectory', normalizes
        them so that the sum of shift scores is 1 or -1. The trajectory
        normalization cannot be applied if the total shift score is 0, so
        scores are left unnormalized if the total is 0 and 'trajectory' is
        specified
    """
    def __init__(self, type2freq_1, type2freq_2, type2score_1=None,
                 type2score_2=None, reference_value=None, stop_lens=None,
                 normalization='variation'):
        # Set type2freq dictionaries
        self.type2freq_1 = type2freq_1.copy()
        self.type2freq_2 = type2freq_2.copy()
        # Set type2score dictionaries
        if type2score_1 is not None and type2score_2 is not None:
            self.type2score_1,lex_ref = get_score_dictionary(type2score_1)
            self.type2score_2,_ = get_score_dictionary(type2score_2)
            if type2score_1 != type2score_2:
                self.show_score_diffs = True
            else:
                self.show_score_diffs = False
        elif type2score_1 is not None:
            self.type2score_1,lex_ref = get_score_dictionary(type2score_1)
            self.type2score_2 = self.type2score_1
            self.show_score_diffs = False
        elif type2score_2 is not None:
            self.type2score_2,lex_ref = get_score_dictionary(type2score_2)
            self.type2score_1 = self.type2score_2
            self.show_score_diffs = False
        else:
            self.type2score_1 = {t : 1 for t in self.type2freq_1}
            self.type2score_2 = {t : 1 for t in self.type2freq_2}
            self.show_score_diffs = False
        # Filter type dictionaries by stop lense
        self.stop_lens = stop_lens
        if stop_lens is not None:
            self.type2freq_1,self.type2score_1,sw_1 = filter_by_scores(self.type2freq_1,
                                                                       self.type2score_1,
                                                                       stop_lens)
            self.type2freq_2,self.type2score_2,sw_2 = filter_by_scores(self.type2freq_2,
                                                                       self.type2score_2,
                                                                       stop_lens)
            self.stop_words = sw_1.union(sw_2)
        # Get common vocabulary
        self.types = get_types(self.type2freq_1, self.type2score_1,
                               self.type2freq_2, self.type2score_2)
        # Assume missing scores in each vocabulary (TODO: add options)
        missing_scores_info = get_missing_scores(self.type2score_1, self.type2score_2)
        self.type2score_1 = missing_scores_info[0]
        self.type2score_2 = missing_scores_info[1]
        self.missing_score_types = missing_scores_info[2]
        #self.missing_score_types = set()

        # Set reference value
        if reference_value is not None:
            if reference_value == 'average':
                self.reference_value = self.get_weighted_score(self.type2freq_1,
                                                               self.type2score_1)
            else:
                self.reference_value = reference_value
        else:
            if lex_ref is not None:
                self.reference_value = lex_ref
            else:
                self.reference_value = 0

        # Get shift scores
        self.normalization = normalization
        self.get_shift_scores(details=False)

    def get_weighted_score(self, type2freq, type2score):
        """
        Calculate an average score according to a set of frequencies and scores

        Parameters
        ----------
        type2freq: dict
            Keys are types and values are frequencies
        type2score: dict
            Keys are types and values are scores

        Returns
        -------
        s_avg: float
            Average weighted score of system
        """
        # Check we have a vocabulary to work with
        types = set(type2freq.keys()).intersection(set(type2score.keys()))
        if len(types) == 0:
            return
        # Get weighted score and total frequency
        f_total = sum([freq for t,freq in type2freq.items() if t in types])
        s_weighted = sum([type2score[t] * freq for t,freq in type2freq.items()
                          if t in types])
        s_avg = s_weighted / f_total
        return s_avg

    def get_shift_scores(self, details=False):
        """
        Calculates the type shift scores between the two systems

        Parameters
        ----------
        details: boolean
            If true, returns each of the major components of each type's shift
            score, along with the overall shift scores. Otherwise, only returns
            the overall shift scores

        Returns
        -------
        type2p_diff: dict
            If details is True, returns dict where keys are types and values are
            the difference in relatively frequency, i.e. p_i,2 - p_i,1 for type i
        type2s_diff: dict,
            If details is True, returns dict where keys are types and values are
            the relative differences in score, i.e. s_i,2 - s_i,1 for type i
        type2p_avg: dict,
            If details is True, returns dict where keys are types and values are
            the average relative frequencies, i.e. 0.5*(p_i,1+p_i,2) for type i
        type2s_ref_diff: dict
            If details is True, returns dict where keys are types and values are
            relative deviation from reference score, i.e. 0.5*(s_i,2+s_i,1)-s_ref
            for type i
        type2shift_score: dict
            Keys are types and values are shift scores. The overall shift scores
            are normalized according to the `normalization` parameter of the
            Shift object
        """
        s_avg_ref = self.reference_value

        # Get total frequencies
        total_freq_1 = sum([freq for t,freq in self.type2freq_1.items() if t in self.types])
        total_freq_2 = sum([freq for t,freq in self.type2freq_2.items() if t in self.types])
        # Get relative frequency of types in both systems
        type2p_1 = {t:self.type2freq_1[t] / total_freq_1 if t in self.type2freq_1 else 0
                    for t in self.types}
        type2p_2 = {t:self.type2freq_2[t] / total_freq_2 if t in self.type2freq_2 else 0
                    for t in self.types}

        # Calculate shift components
        type2p_avg = dict()
        type2p_diff = dict()
        type2s_diff = dict()
        type2s_ref_diff = dict()
        type2shift_score = dict()
        for t in self.types:
            type2p_avg[t] = 0.5 * (type2p_1[t] + type2p_2[t])
            type2p_diff[t] = type2p_2[t] - type2p_1[t]
            type2s_diff[t] = self.type2score_2[t] - self.type2score_1[t]
            type2s_ref_diff[t] = 0.5 * (self.type2score_2[t] + self.type2score_1[t])\
                                 - s_avg_ref
            type2shift_score[t] = type2p_diff[t] * type2s_ref_diff[t]\
                                 + type2s_diff[t] * type2p_avg[t]

        # Normalize the total shift scores
        total_diff = sum(type2shift_score.values())
        self.diff = total_diff
        if self.normalization == 'variation':
            abs_sum = sum(abs(s) for s in type2shift_score.values())
            self.norm = abs_sum
        elif self.normalization == 'trajectory' and total_diff != 0:
            self.norm = abs(total_diff)
        else:
            self.norm = 1
        type2shift_score = {t : shift_score / self.norm for t,shift_score
                            in type2shift_score.items()}

        # Set results in shift object
        self.type2p_diff = type2p_diff
        self.type2s_diff = type2s_diff
        self.type2p_avg = type2p_avg
        self.type2s_ref_diff = type2s_ref_diff
        self.type2shift_score = type2shift_score
        # Return shift scores
        if details:
            return type2p_diff,type2s_diff,type2p_avg,type2s_ref_diff,type2shift_score
        else:
            return type2shift_score

    def get_shift_component_sums(self):
        """
        Calculates the cumulative contribution of each component of the different
        kinds of shift scores.

        Returns
        -------
        Dictionary with six keys, one for each of the different component
        contributions: pos_s_pos_p, pos_s_neg_p, neg_s_pos_p, neg_s_neg_p,
        pos_s, neg_s. Values are the total contribution from that component
        across all types
        """
        # Get shift scores
        if self.type2shift_score is None:
            shift_scores = self.get_shift_scores(details=True)
        else:
            shift_scores = [(t, self.type2p_diff[t], self.type2s_diff[t],
                             self.type2p_avg[t], self.type2s_ref_diff[t],
                             self.type2shift_score[t]) for t in self.type2s_diff]

        # Sum up components of shift score
        pos_s_pos_p = 0
        pos_s_neg_p = 0
        neg_s_pos_p = 0
        neg_s_neg_p = 0
        pos_s = 0
        neg_s = 0
        for t,p_diff,s_diff,p_avg,s_ref_diff, _ in shift_scores:
            # Get contribution of p_diff*s_ref_diff term
            if s_ref_diff > 0:
                if p_diff > 0:
                    pos_s_pos_p += p_diff * s_ref_diff
                else:
                    pos_s_neg_p += p_diff * s_ref_diff
            else:
                if p_diff > 0:
                    neg_s_pos_p += p_diff * s_ref_diff
                else:
                    neg_s_neg_p += p_diff * s_ref_diff
            # Get contribution of s_diff term
            if s_diff > 0:
                pos_s += p_avg * s_diff
            else:
                neg_s += p_avg * s_diff

        return {'pos_s_pos_p': pos_s_pos_p, 'pos_s_neg_p': pos_s_neg_p,
                'neg_s_pos_p': neg_s_pos_p, 'neg_s_neg_p': neg_s_neg_p,
                'pos_s': pos_s, 'neg_s': neg_s}

    def get_shift_graph(self, top_n=50, text_size_inset=True,
                        cumulative_inset=True, show_plot=True, filename=None,
                        **kwargs):
        """
        Plot the shift graph between two systems of types

        Parameters
        ----------
        top_n: int, optional
            Display the top_n types as sorted by their absolute contribution to
            the difference between systems
        cumulative_inset: bool, optional
            Whether to show an inset showing the cumulative contributions to the
            shift by ranked types
        text_size_inset: bool, optional
            Whether to show an inset showing the relative sizes of each system
        show_plot: bool, optional
            Whether to show plot when it is done being rendered
        filename: str, optional
            If not None, name of the file for saving the shift graph

        Returns
        -------
        ax
            Matplotlib ax of shift graph. Displays shift graph if show_plot=True
        """
        # Set plotting parameters
        kwargs = get_plot_params(kwargs, self.show_score_diffs)

        # Get type score components
        type_scores = [(t, self.type2p_diff[t], self.type2s_diff[t],
                        self.type2p_avg[t], self.type2s_ref_diff[t],
                        self.type2shift_score[t]) for t in self.type2s_diff]
        # Reverse sorting to get highest scores, then reverse top n for plotting
        type_scores = sorted(type_scores, key=lambda x:abs(x[-1]),
                             reverse=True)[:top_n]
        type_scores.reverse()

        # Get bar heights and colors
        bar_dims = get_bar_dims(type_scores, self.norm, kwargs)
        bar_colors = get_bar_colors(type_scores, kwargs)

        # Initialize plot
        f,ax = plt.subplots(figsize=(kwargs['width'], kwargs['height']))
        ax.margins(kwargs['y_margin'])
        # Plot type contributions
        ax = plot_contributions(ax, top_n, bar_dims, bar_colors, kwargs)
        # Plot total sum contributions
        total_comp_sums = self.get_shift_component_sums()
        bar_order = get_bar_order(kwargs)
        ax,comp_bar_heights,bar_order = plot_total_contribution_sums(ax, total_comp_sums,
                                                                     bar_order, top_n,
                                                                     bar_dims, kwargs)
        # Get labels for bars
        type_labels = [t for (t,_,_,_,_,_) in type_scores]
        # Add indicator if type borrwed a score
        m_sym = kwargs['missing_symbol']
        type_labels = [t + m_sym if t in self.missing_score_types else t
                       for t in type_labels]
        # Get labels for total contribution bars
        bar_labels = [kwargs['symbols'][b] for b in bar_order]
        labels = type_labels + bar_labels
        # Set font type
        if kwargs['serif']:
            set_serif()
        # Set labels
        if kwargs['detailed']:
            ax = set_bar_labels(f, ax, top_n, labels, bar_dims['label_heights'],
                                comp_bar_heights, kwargs)
        else:
            ax = set_bar_labels(f, ax, top_n, labels, bar_dims['total_heights'],
                                comp_bar_heights, kwargs)

        # Add center dividing line
        ax.axvline(0, ls='-', color='black', lw=1.0, zorder=20)

        # Add dividing line between types and component bars
        ax.axhline(top_n+1, ls='-', color='black', lw=0.7, zorder=20)
        if kwargs['show_total']:
            ax.axhline(top_n+2.75, ls='-', color='black', lw=0.5, zorder=20)

        # Set insets
        if cumulative_inset:
            f = get_cumulative_inset(f, self.type2shift_score, top_n,
                                     self.normalization, kwargs)
        if text_size_inset:
            f = get_text_size_inset(f, self.type2freq_1, self.type2freq_2, kwargs)

        # Make x-tick labels bigger, flip y-axis ticks and label every 5th one
        ax = set_ticks(ax, top_n, kwargs)

        # Set axis spines
        ax = set_spines(ax, kwargs)

        # Set axis labels and title
        ax.set_xlabel(kwargs['xlabel'], fontsize=kwargs['xlabel_fontsize'])
        ax.set_ylabel(kwargs['ylabel'], fontsize=kwargs['ylabel_fontsize'])
        if 'title' not in kwargs:
            s_avg_1 = self.get_weighted_score(self.type2freq_1,self.type2score_1)
            s_avg_2 = self.get_weighted_score(self.type2freq_2,self.type2score_2)
            title = '{}: '.format(kwargs['system_names'][0])\
                    + r'$\Phi_{avg}=$'+'{0:.2f}'.format(s_avg_1)+'\n'\
                    +'{}: '.format(kwargs['system_names'][1])\
                    + r'$\Phi_{avg}=$'+'{0:.2f}'.format(s_avg_2)
            kwargs['title'] = title
        ax.set_title(kwargs['title'], fontsize=kwargs['title_fontsize'])

        # Show and return plot
        if kwargs['tight']:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=kwargs['dpi'])
        if show_plot:
            plt.show()
        return ax
