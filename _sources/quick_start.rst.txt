.. _quick_start:

###############
Getting Started
###############

Installation
============

From PyPI (Recommended)
-----------------------

Once available on PyPI, install scikit-bayes with pip:

.. prompt:: bash $

    pip install scikit-bayes

From Source
-----------

To install the development version directly from GitHub:

.. prompt:: bash $

    pip install git+https://github.com/ptorrijos99/scikit-bayes.git


Quick Examples
==============

MixedNB: Handling Mixed Data Types
----------------------------------

:class:`skbayes.MixedNB` automatically detects and handles datasets with 
Gaussian (continuous), Categorical, and Bernoulli (binary) features:

.. code-block:: python

    import numpy as np
    from skbn import MixedNB

    # Data: [Gaussian, Bernoulli, Categorical]
    X = np.array([
        [0.5, 0, 0],   # continuous, binary, categories 0-2
        [-1.2, 1, 1],
        [0.6, 1, 2],
        [-0.1, 0, 0],
    ])
    y = np.array([0, 1, 1, 0])

    clf = MixedNB()
    clf.fit(X, y)
    print(clf.predict([[-0.5, 1, 1]]))  # Output: [1]


AnDE: Solving Problems Naive Bayes Cannot
-----------------------------------------

:class:`skbayes.AnDE` relaxes the independence assumption of Naive Bayes,
allowing it to capture feature dependencies like the XOR problem:

.. code-block:: python

    import numpy as np
    from skbayes import AnDE

    # XOR problem: class depends on interaction of features
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR: same sign → 0, different → 1

    # Naive Bayes fails (~50% accuracy), AnDE succeeds
    clf = AnDE(n_dependence=1, n_bins=2)
    clf.fit(X, y)
    print(clf.predict(X))  # Output: [0, 1, 1, 0] ✓


Development Setup
=================

This project uses `pixi <https://pixi.sh>`_ for environment management.

Install pixi
------------

Follow the instructions at https://pixi.sh/latest/#installation

Common Commands
---------------

Run tests:

.. prompt:: bash $

    pixi run test

Run linter:

.. prompt:: bash $

    pixi run lint

Build documentation:

.. prompt:: bash $

    pixi run build-doc

Activate development environment:

.. prompt:: bash $

    pixi shell -e dev

This activates an environment with all dependencies for testing, linting,
and building documentation.


What's Next?
============

* :ref:`User Guide <user_guide>` - Detailed documentation of all estimators
* :ref:`API Reference <api>` - Complete API documentation  
* :ref:`Examples <general_examples>` - Gallery of usage examples
