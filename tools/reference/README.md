# External reference fixtures

`skbn/tests/test_external_references.py` checks scikit-bayes against other,
independent implementations. Their outputs are frozen in `skbn/tests/data/`,
so the tests need neither Java nor the other packages. This folder holds the
generator of those files.

| File | Contents |
| --- | --- |
| `make_reference.py` | Builds the datasets, writes them to ARFF, runs Weka, pyAgrum and pgmpy, and writes the JSON fixtures. |
| `WekaReference.java` | Command-line driver of the Weka classifiers and filter; prints their outputs as text lines. |
| `pom.xml` | The Maven coordinates of the Weka jars. |

## Software and options

| Reference | Version (Maven coordinates / PyPI) | Options |
| --- | --- | --- |
| Weka `NaiveBayes` | `nz.ac.waikato.cms.weka:weka-stable:3.8.6` | defaults (normal estimator, no `-K`, no `-D`) |
| Weka `BayesNet` + `TAN` | same | `-D -Q weka.classifiers.bayes.net.search.local.TAN -- -S ENTROPY -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 1.0` (and `-A 0.5`) |
| Weka `AODE` | `nz.ac.waikato.cms.weka:averagedOneDependenceEstimators:1.2.1` (run on weka-stable 3.8.6) | `-F 1`, and `-F 100`, `-F 150` on the synthetic data; Laplace estimates (no `-M`) |
| Weka supervised `Discretize` | `weka-stable:3.8.6` | defaults: Fayyad and Irani's MDL (no `-K`), no better encoding (no `-E`) |
| pgmpy `TreeSearch` | `pgmpy==1.1.2` | `estimator_type="tan"`, `root_node="x0"`, mutual information weights |
| pyAgrum `BNClassifier` | `pyagrum==3.3.2` | `learningMethod="TAN"`, `prior="NoPrior"` |

The fixtures were produced with OpenJDK 21, Python 3.13, NumPy 2.5.3,
scikit-learn 1.9.1, pandas 3.1.1 and networkx 3.7; the versions are also
recorded in the `meta` entry of every JSON file.

TAN is run with `-S ENTROPY`, whose score gain for a parent is N times the
conditional mutual information, the criterion of Friedman et al. (1997) used
by `skbn.TAN`. Weka's default, `-S BAYES`, scores the edges with a Bayesian
(Dirichlet) metric and learns a different tree on all four datasets here.

pyAgrum's `TAN` is not a maximum spanning tree: it adds the edges in
decreasing conditional mutual information while they close no cycle and their
weight is at least `sum_{i<j} I_ij / (d (d - 1))`, so it keeps the part of the
tree above that threshold. Without a prior it then fails to estimate the
parameters of a parent configuration that never occurs; the generator reads
the arcs, which are added before that step.

## Datasets

All are built deterministically in `make_reference.py` and stored, integer
coded, in `reference_datasets.json`, so the tests do not depend on the
discretizers of the installed scikit-learn:

- `iris_q3`, `wine_q4`, `breast_cancer_q3`: scikit-learn's datasets, split with
  a stratified `train_test_split(random_state=0)` and discretized once with
  `KBinsDiscretizer(encode="ordinal", strategy="quantile")` fitted on the
  training part (3, 4 and 3 bins). Iris keeps 30 training samples per class;
  breast cancer uses 300 of the samples and ten attributes (see the comment
  in `build_datasets` for why mean perimeter and area are left out).
- `synthetic`: six attributes with 2 to 6 values, four unbalanced classes,
  dependencies between attributes and rare values.
- `gaussian_grid`: three integer-valued Gaussian attributes whose training
  values cover every integer of their range (Weka's numeric precision is then
  exactly 1), and `iris_numeric`, the raw iris measurements.

Every value of a test row and every value declared in the ARFF header occurs
in training. The MDL cut-points are computed on the full iris, wine,
breast cancer and digits datasets, which the tests load from scikit-learn.

## Regenerating

Java 11 or later, Maven and Python with scikit-bayes are needed; pyAgrum and
pgmpy are installed in a separate folder so that they do not change the
environment:

```bash
REF=/tmp/skbn-reference
mvn -q -B -f tools/reference/pom.xml dependency:copy-dependencies -DoutputDirectory=$REF/lib
javac -cp "$REF/lib/*" -d $REF/classes tools/reference/WekaReference.java
pip install --target $REF/pylibs pyagrum==3.3.2 pgmpy==1.1.2
# Keep numpy, scipy and scikit-learn of the main environment
rm -rf $REF/pylibs/numpy* $REF/pylibs/scipy* $REF/pylibs/sklearn $REF/pylibs/scikit_learn*
PYTHONPATH=$PWD:$REF/pylibs python tools/reference/make_reference.py \
    --classpath "$REF/lib/*:$REF/classes" --workdir $REF/work
```

The script overwrites `skbn/tests/data/reference_*.json`; run
`pytest skbn/tests/test_external_references.py` afterwards.
