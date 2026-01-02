.. _general_examples:

Example Gallery
===============

This gallery contains examples demonstrating the capabilities of scikit-bayes.
All examples use matplotlib for visualization and can be downloaded as Python
scripts or Jupyter notebooks.


MixedNB Examples
----------------

Examples demonstrating the Mixed Naive Bayes classifier for heterogeneous data.

.. include:: plot_gaussiannb_equivalence.py
   :start-line: 1
   :end-line: 13

.. include:: plot_categoricalnb_equivalence.py
   :start-line: 1
   :end-line: 13

.. include:: plot_bernoullinb_equivalence.py
   :start-line: 1
   :end-line: 13

.. include:: plot_mixednb_vs_pipeline.py
   :start-line: 1
   :end-line: 24


AnDE Examples
-------------

Averaged n-Dependence Estimators solving problems Naive Bayes cannot.

.. include:: plot_aode_xor.py
   :start-line: 1
   :end-line: 26

.. include:: plot_a2de_xor_3d.py
   :start-line: 1
   :end-line: 13

.. include:: plot_a2de_xor_slice.py
   :start-line: 1
   :end-line: 13

.. include:: plot_aode_mixed_demo.py
   :start-line: 1
   :end-line: 13


Benchmarks and Comparisons
--------------------------

Performance comparisons of different model variants.

.. include:: compare_ande_variants.py
   :start-line: 1
   :end-line: 19

.. include:: compare_ande_interpretability.py
   :start-line: 1
   :end-line: 13
