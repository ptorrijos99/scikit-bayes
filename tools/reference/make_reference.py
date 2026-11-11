"""Generate the external reference fixtures of
``skbn/tests/test_external_references.py``.

The script builds small integer-coded datasets, writes them to ARFF, runs Weka
on them through ``WekaReference.java`` and pyAgrum and pgmpy in this process,
and freezes their inputs and outputs as JSON files in ``skbn/tests/data/``.
See README.md in this folder for the versions and the commands.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "skbn" / "tests" / "data"

WEKA = {
    "group": "nz.ac.waikato.cms.weka",
    "artifact": "weka-stable",
    "version": "3.8.6",
}
AODE_PKG = {
    "group": "nz.ac.waikato.cms.weka",
    "artifact": "averagedOneDependenceEstimators",
    "version": "1.2.1",
}

# --- Datasets ----------------------------------------------------------------


def _recode(X_train, X_test):
    """Codes 0..r-1 of the training values of every column; columns with a
    single training value are dropped, and so are the test rows holding a value
    never seen in training (an unseen value has no Weka counterpart: skbn
    marginalises it, Weka gives it the Laplace estimate of an empty cell)."""
    keep = [j for j in range(X_train.shape[1]) if len(np.unique(X_train[:, j])) > 1]
    tr, te = [], []
    seen = np.ones(len(X_test), dtype=bool)
    for j in keep:
        values = np.unique(X_train[:, j])
        tr.append(np.searchsorted(values, X_train[:, j]))
        pos = np.minimum(np.searchsorted(values, X_test[:, j]), len(values) - 1)
        seen &= values[pos] == X_test[:, j]
        te.append(pos)
    return np.column_stack(tr), np.column_stack(te), seen


def _quantile_dataset(loader, n_bins, test_size, seed=0, features=None, n_rows=None):
    X, y = loader(return_X_y=True)
    if features is not None:
        X = X[:, features]
    if n_rows is not None:
        X, _, y, _ = train_test_split(
            X, y, train_size=n_rows, stratify=y, random_state=seed
        )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    D_tr = disc.fit_transform(X_tr).astype(int)
    D_te = disc.transform(X_te).astype(int)
    D_tr, D_te, seen = _recode(D_tr, D_te)
    return D_tr, y_tr, D_te[seen], y_te[seen]


def _synthetic_categorical(seed=0, n_train=300, n_test=90):
    """Six attributes with cardinalities 2..6, four unbalanced classes,
    dependencies between attributes and some rare values and combinations."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    y = rng.choice(4, size=n, p=[0.45, 0.3, 0.17, 0.08])
    x0 = (rng.rand(n) < 0.25 + 0.15 * y).astype(int)  # 2 values
    x1 = (x0 + y + rng.choice(3, size=n, p=[0.7, 0.2, 0.1])) % 3  # 3 values
    x2 = np.where(rng.rand(n) < 0.6, (x1 + y) % 4, rng.randint(0, 4, n))  # 4 values
    # 5 values, the last two rare
    x3 = rng.choice(5, size=n, p=[0.4, 0.3, 0.22, 0.05, 0.03])
    x3 = np.where((y == 3) & (rng.rand(n) < 0.3), 4, x3)
    x4 = rng.randint(0, 3, n)  # noise, 3 values
    x5 = (x3 + 2 * y + rng.choice(6, size=n, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])) % 6
    X = np.column_stack([x0, x1, x2, x3, x4, x5])
    X_tr, X_te, seen = _recode(X[:n_train], X[n_train:])
    return X_tr, y[:n_train], X_te[seen], y[n_train:][seen]


def _gaussian_grid(seed=0, n_train=240, n_test=60):
    """Integer-valued Gaussian attributes whose training values cover every
    integer between their minimum and maximum, so that Weka's numeric precision
    is exactly 1 and its rounding leaves the values unchanged."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    y = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    means = np.array([[40.0, 60.0, 50.0], [45.0, 52.0, 58.0], [50.0, 50.0, 44.0]])
    sds = np.array([[6.0, 5.0, 7.0], [8.0, 6.0, 5.0], [7.0, 9.0, 6.0]])
    X = np.rint(means[y] + sds[y] * rng.randn(n, 3)).astype(int)
    for j in range(3):
        # Clip to the run of consecutive training values around the median
        values = np.unique(X[:n_train, j])
        mid = np.searchsorted(values, np.median(X[:n_train, j]))
        lo = hi = mid
        while lo > 0 and values[lo - 1] == values[lo] - 1:
            lo -= 1
        while hi < len(values) - 1 and values[hi + 1] == values[hi] + 1:
            hi += 1
        X[:, j] = np.clip(X[:, j], values[lo], values[hi])
    X_tr, X_te = X[:n_train], X[n_train:]
    for j in range(3):
        values = np.unique(X_tr[:, j])
        assert np.array_equal(values, np.arange(values[0], values[-1] + 1)), j
    return X_tr, y[:n_train], X_te, y[n_train:]


def _iris_numeric(seed=0):
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=60, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_te, y_te


def build_datasets():
    """Integer-coded nominal datasets and two numeric ones."""
    nominal = {
        # 30 training samples per class: equal class priors in Weka and skbn
        "iris_q3": _quantile_dataset(load_iris, 3, test_size=60),
        "wine_q4": _quantile_dataset(load_wine, 4, test_size=0.3),
        # Ten attributes of 300 of the samples. Mean perimeter and area are
        # left out: their quantile bins nearly coincide with those of the
        # mean radius, which ties the conditional mutual information of two
        # edges in exact arithmetic, and the maximum spanning tree is then
        # not unique (each implementation breaks the tie its own way)
        "breast_cancer_q3": _quantile_dataset(
            load_breast_cancer,
            3,
            test_size=0.3,
            features=[0, 1, 4, 5, 6, 7, 8, 9, 20, 21],
            n_rows=300,
        ),
        "synthetic": _synthetic_categorical(),
    }
    numeric = {"gaussian_grid": _gaussian_grid(), "iris_numeric": _iris_numeric()}
    return nominal, numeric


# --- ARFF --------------------------------------------------------------------


def write_arff(path, X, y, n_classes, cards=None):
    """ARFF file with the class last. With ``cards`` the attributes are
    nominal with the values ``0..cards[j]-1``; otherwise numeric, written
    with ``repr`` (the shortest string that reads back to the same double)."""
    lines = [f"@relation {path.stem}"]
    for j in range(X.shape[1]):
        if cards is None:
            lines.append(f"@attribute a{j} numeric")
        else:
            values = ",".join(str(v) for v in range(cards[j]))
            lines.append(f"@attribute a{j} {{{values}}}")
    classes = ",".join(str(c) for c in range(n_classes))
    lines += [f"@attribute class {{{classes}}}", "@data"]
    for row, label in zip(X, y):
        values = [str(int(v)) if cards is not None else repr(float(v)) for v in row]
        lines.append(",".join(values + [str(int(label))]))
    path.write_text("\n".join(lines) + "\n")


# --- Weka --------------------------------------------------------------------


class Weka:
    def __init__(self, classpath):
        self.classpath = classpath
        self.version = None

    def run(self, *args):
        out = subprocess.run(
            ["java", "-cp", self.classpath, "WekaReference", *map(str, args)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result = {"proba": [], "normal": [], "parents": {}, "cuts": {}}
        for line in out.splitlines():
            key, _, rest = line.partition(" ")
            if key == "weka_version":
                self.version = rest
            elif key == "options":
                result["options"] = rest
            elif key == "proba":
                result["proba"].append([float(v) for v in rest.split()])
            elif key == "normal":
                a, c, mean, std, prec = rest.split()
                result["normal"].append(
                    [int(a), int(c), float(mean), float(std), float(prec)]
                )
            elif key == "parents":
                a, *p = rest.split()
                result["parents"][int(a)] = [int(v) for v in p]
            elif key == "cuts":
                a, *c = rest.split()
                result["cuts"][int(a)] = [float(v) for v in c]
        return result


# --- pyAgrum and pgmpy (TAN structure) ---------------------------------------


def pyagrum_tan(X, y):
    """Undirected edges of pyAgrum's BNClassifier(learningMethod='TAN') with no
    prior on the counts. The values are passed as strings so that pyAgrum keeps
    them as labels instead of discretizing them."""
    import pyagrum
    from pyagrum.lib.discreteTypeProcessor import DiscreteTypeProcessor
    from pyagrum.skbn import BNClassifier

    clf = BNClassifier(
        type_processor=DiscreteTypeProcessor(),
        learningMethod="TAN",
        prior="NoPrior",
    )
    Xs = np.array([[f"v{v}" for v in row] for row in X], dtype=object)
    try:
        clf.fit(Xs, np.array([f"c{v}" for v in y], dtype=object))
    except pyagrum.DatabaseError:
        # Without a prior, pyAgrum refuses to estimate the parameters of a
        # parent configuration that never occurs; the arcs, which are all we
        # read, are already in clf.bn_ (_fitTAN adds them before estimating)
        pass
    bn = clf.bn_
    target = bn.idFromName(clf.target_)
    edges = []
    for tail, head in bn.arcs():
        if target in (tail, head):
            continue
        a, b = (int(bn.variable(t).name()[1:]) for t in (tail, head))
        edges.append(sorted([a, b]))
    return sorted(edges)


def pgmpy_tan(X, y, root=0):
    """Undirected edges of pgmpy's TreeSearch(estimator_type='tan') rooted at
    attribute ``root``, and its conditional mutual information matrix."""
    import pandas as pd
    from pgmpy.estimators import TreeSearch

    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    df["class"] = y
    dag = TreeSearch(df, root_node=f"x{root}", n_jobs=1).estimate(
        estimator_type="tan", class_node="class", show_progress=False
    )
    edges = sorted(
        sorted([int(u[1:]), int(v[1:])])
        for u, v in dag.edges()
        if "class" not in (u, v)
    )
    w = TreeSearch._get_conditional_weights(df, "class", n_jobs=1, show_progress=False)
    d = X.shape[1]
    return edges, np.asarray(w)[:d, :d].tolist()


# --- Main --------------------------------------------------------------------


def _dataset_entry(X_tr, y_tr, X_te, y_te, nominal):
    conv = (lambda a: a.astype(int).tolist()) if nominal else (lambda a: a.tolist())
    entry = {
        "X_train": conv(X_tr),
        "y_train": y_tr.astype(int).tolist(),
        "X_test": conv(X_te),
        "y_test": y_te.astype(int).tolist(),
    }
    if nominal:
        entry["cards"] = (X_tr.max(axis=0) + 1).tolist()
    return entry


def _json(obj, indent=0):
    """JSON with one line per list of numbers (a row, a posterior), so that
    the files stay small and their diffs readable."""
    pad = " " * indent
    if isinstance(obj, dict):
        items = [
            f"{pad} {json.dumps(str(k))}: {_json(v, indent + 1).lstrip()}"
            for k, v in obj.items()
        ]
        return pad + "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, dict)):
        rows = [_json(v, indent + 1) for v in obj]
        return pad + "[\n" + ",\n".join(rows) + "\n" + pad + "]"
    return pad + json.dumps(obj, separators=(",", ":"))


def dump(name, payload):
    path = DATA_DIR / name
    path.write_text(_json(payload) + "\n")
    # The custom layout must hold exactly the payload
    assert json.loads(path.read_text()) == json.loads(json.dumps(payload))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classpath", required=True, help="Weka jars and WekaReference"
    )
    parser.add_argument("--workdir", required=True, help="where the ARFF files go")
    args = parser.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    weka = Weka(args.classpath)

    nominal, numeric = build_datasets()
    datasets = {}
    arff = {}
    for name, (X_tr, y_tr, X_te, y_te) in nominal.items():
        datasets[name] = _dataset_entry(X_tr, y_tr, X_te, y_te, nominal=True)
        cards = datasets[name]["cards"]
        k = int(y_tr.max()) + 1
        arff[name] = (work / f"{name}_train.arff", work / f"{name}_test.arff")
        write_arff(arff[name][0], X_tr, y_tr, k, cards)
        write_arff(arff[name][1], X_te, y_te, k, cards)
    for name, (X_tr, y_tr, X_te, y_te) in numeric.items():
        datasets[name] = _dataset_entry(X_tr, y_tr, X_te, y_te, nominal=False)
        k = int(y_tr.max()) + 1
        arff[name] = (work / f"{name}_train.arff", work / f"{name}_test.arff")
        write_arff(arff[name][0], X_tr, y_tr, k)
        write_arff(arff[name][1], X_te, y_te, k)

    # Naive Bayes on every dataset
    nb = {name: weka.run("nb", *arff[name]) for name in datasets}

    # TAN: CMI tree (-S ENTROPY) at two smoothing levels
    tan = {
        name: {str(a): weka.run("tan", *arff[name], a, "ENTROPY") for a in (1.0, 0.5)}
        for name in nominal
    }
    # Weka's default score (-S BAYES) is not the conditional mutual
    # information: report whether its tree differs, for the README
    for name in nominal:
        bayes = weka.run("tan", *arff[name], 0.5, "BAYES")["parents"]
        same = bayes == tan[name]["1.0"]["parents"]
        print(
            f"TAN {name}: -S BAYES tree {'equals' if same else 'differs from'} -S ENTROPY"
        )

    # AODE with the default frequency limit, and higher limits on the
    # synthetic data (rare values) to exclude parents
    aode = {name: {"1": weka.run("aode", *arff[name], 1)} for name in nominal}
    for f in (100, 150):
        aode["synthetic"][str(f)] = weka.run("aode", *arff["synthetic"], f)

    # MDL discretization on the full numeric datasets (digits: integer values
    # with many ties)
    mdl = {}
    for name, loader in [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ]:
        X, y = loader(return_X_y=True)
        path = work / f"{name}_full.arff"
        write_arff(path, X, y, int(y.max()) + 1)
        out = weka.run("mdl", path)
        mdl[name] = {
            "options": out["options"],
            "cut_points": [out["cuts"][j] for j in range(X.shape[1])],
        }

    # TAN structure from pyAgrum and pgmpy
    other = {}
    for name in nominal:
        d = datasets[name]
        X, y = np.array(d["X_train"]), np.array(d["y_train"])
        edges, cmi = pgmpy_tan(X, y)
        w = np.sort(np.asarray(cmi)[np.triu_indices(X.shape[1], 1)])
        if np.any(np.diff(w) <= 1e-12 * w[1:]):
            print(f"WARNING {name}: tied CMI weights, the TAN tree may not be unique")
        other[name] = {
            "pyagrum_edges": pyagrum_tan(X, y),
            "pgmpy_edges": edges,
            "pgmpy_cmi": cmi,
        }

    meta = {
        "weka": f"{WEKA['group']}:{WEKA['artifact']}:{WEKA['version']}",
        "weka_version": weka.version,
        "aode_package": f"{AODE_PKG['group']}:{AODE_PKG['artifact']}:{AODE_PKG['version']}",
        "java": subprocess.run(
            ["java", "-version"], capture_output=True, text=True
        ).stderr.splitlines()[-3:],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "generator": "tools/reference/make_reference.py",
    }
    dump("reference_datasets.json", {"meta": meta, "datasets": datasets})
    dump(
        "reference_weka.json",
        {
            "meta": meta,
            "naive_bayes": {
                n: {k: r[k] for k in ("options", "proba", "normal")}
                for n, r in nb.items()
            },
            "tan": {
                n: {
                    "options": {a: r["options"] for a, r in t.items()},
                    # Attribute parent of every attribute, -1 for the root
                    "parents": [
                        (p[0] if p else -1)
                        for _, p in sorted(t["1.0"]["parents"].items())
                    ],
                    "proba": {a: r["proba"] for a, r in t.items()},
                }
                for n, t in tan.items()
            },
            "aode": {
                n: {
                    f: {"options": r["options"], "proba": r["proba"]}
                    for f, r in a.items()
                }
                for n, a in aode.items()
            },
            "mdl": mdl,
        },
    )
    dump(
        "reference_tan_structure.json",
        {
            "meta": {
                "pyagrum": version("pyagrum"),
                "pgmpy": version("pgmpy"),
                "pandas": version("pandas"),
                "networkx": version("networkx"),
                "python": platform.python_version(),
            },
            "tan": other,
        },
    )


if __name__ == "__main__":
    main()
