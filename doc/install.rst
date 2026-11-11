.. _install:

############
Installation
############

scikit-bayes needs Python 3.10 or later, NumPy, SciPy, joblib, Numba and
scikit-learn 1.6 or later. pip installs them with the package.

From PyPI
=========

.. prompt:: bash $

    pip install scikit-bayes

From GitHub
===========

The development version, from the ``main`` branch:

.. prompt:: bash $

    pip install git+https://github.com/ptorrijos99/scikit-bayes.git

From source
===========

To work on the code, clone the repository and install it in editable mode,
so that changes to the sources take effect without reinstalling:

.. prompt:: bash $

    git clone https://github.com/ptorrijos99/scikit-bayes.git
    cd scikit-bayes
    pip install -e .

The development environments, with the dependencies of the tests, the linter
and the documentation, are managed with `pixi <https://pixi.sh>`_ (see
:ref:`quick_start`).

First use of each model
=======================

The counting and the prediction run in Numba kernels, compiled the first time
each model is used, so the first fit of a model after installing or upgrading
the package takes longer. The compiled code is cached on disk by Numba, and
later sessions load it instead of compiling again (see
:ref:`computational_considerations`). On a laptop the first fit takes from a
couple of seconds (:class:`skbn.MixedNB`) to some ten
(:class:`skbn.AnDE`).

Checking the installation
=========================

.. prompt:: bash $

    python -m skbn

prints the versions of the package and of its dependencies, fits and
evaluates every model once on the iris data, which compiles and caches all
the kernels (about half a minute the first time, a couple of seconds
afterwards), and verifies the results: probabilities that are finite and sum
to one, the equality of :class:`skbn.MixedNB` with scikit-learn's
``GaussianNB`` and the cut-points of :class:`skbn.MDLPDiscretizer`. With
``--n-jobs 4`` it compiles the threaded versions of the kernels. From Python,
:func:`skbn.check_installation` does the same and
:func:`skbn.show_versions` prints the versions alone, to paste in a bug
report.
