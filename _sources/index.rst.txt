.. scikit-bayes documentation master file

:notoc:

################################
scikit-bayes
################################

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/ptorrijos99/scikit-bayes>`__ |
`Issues & Ideas <https://github.com/ptorrijos99/scikit-bayes/issues>`__ |

**scikit-bayes** is a Python package that extends `scikit-learn` with a suite of 
Bayesian Network Classifiers. The primary goal is to provide robust, 
`scikit-learn`-compatible implementations of advanced Bayesian classifiers that
are not available in the core library.

**Key Features:**

* **MixedNB**: Naive Bayes for mixed data types (Gaussian, Categorical, Bernoulli)
* **AnDE Family**: Averaged n-Dependence Estimators that relax the independence assumption
* **Hybrid Models**: ALR and WeightedAnDE for discriminative learning


.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
        :img-top: _static/img/index_getting_started.svg
        :class-card: intro-card
        :shadow: md

        Information about installation and basic usage of scikit-bayes.

        +++

        .. button-ref:: quick_start
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the getting started guideline

    .. grid-item-card::  User guide
        :img-top: _static/img/index_user_guide.svg
        :class-card: intro-card
        :shadow: md

        Learn how to create your own scikit-learn compatible estimators.

        +++

        .. button-ref:: user_guide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card::  API reference
        :img-top: _static/img/index_api.svg
        :class-card: intro-card
        :shadow: md

        Complete reference documentation for all estimators and utilities.

        +++

        .. button-ref:: api
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card::  Examples
        :img-top: _static/img/index_examples.svg
        :class-card: intro-card
        :shadow: md

        Gallery of examples demonstrating AnDE, MixedNB, and more.

        +++

        .. button-ref:: general_examples
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the gallery of examples


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    quick_start
    user_guide
    api
    auto_examples/index
