# Shifterator

> **⚠️ Important: Install from GitHub, not PyPI**
>
> The version of Shifterator on PyPI (`0.3.0`) is outdated and no longer maintained.
> The actively maintained version lives here on GitHub. To install the latest version:
>
> ```bash
> pip install git+https://github.com/ryanjgallagher/shifterator.git
> ```

The Shifterator package provides functionality for constructing **word shift graphs**, vertical bart charts that quantify *which* words contribute to a pairwise difference between two texts and *how* they contribute. By allowing you to look at changes in how words are used, word shifts help you to conduct analyses of sentiment, entropy, and divergence that are fundamentally more interpretable.

<p align="center">
  <img src ="docs/figs/shift_sentiment_detailed_full.png" width="400"/>
</p>


## Install

Python code to produce shift graphs can be installed directly from GitHub:

```bash
pip install git+https://github.com/ryanjgallagher/shifterator.git
```

## Documentation

Documentation is available in the [docs/](docs/) folder of this repository. The Sphinx source can be built locally with `make html` from that directory.

The old ReadTheDocs site (shifterator.readthedocs.io) is no longer maintained and may not reflect the current version.

## Citation

See the following paper for more details on word shifts, and please cite it if you use them in your work:

> Gallagher, R. J., Frank, M. R., Mitchell, Lewis, Schwartz, A. J., Reagan, A. J., Danforth, C. M., Dodds, P. S. (2021). [Generalized Word Shift Graphs: A Method for Visualizing and Explaining Pairwise Comparisons Between Texts](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-021-00260-3). *EPJ Data Science*, 10(4).
