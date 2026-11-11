// Authors: scikit-bayes developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Command-line driver that fits Weka classifiers and filters on ARFF files and
// prints their outputs as plain text lines, for make_reference.py. The class
// is always the last attribute. Numbers are printed with Double.toString, the
// shortest decimal that reads back to the same double.
//
//   nb   TRAIN TEST                 weka.classifiers.bayes.NaiveBayes (defaults)
//   tan  TRAIN TEST ALPHA SCORE     BayesNet + local.TAN (-S SCORE) + SimpleEstimator (-A ALPHA)
//   aode TRAIN TEST F               weka.classifiers.bayes.AODE (-F F, Laplace estimates)
//   mdl  DATA                       weka.filters.supervised.attribute.Discretize (defaults)

import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.AODE;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.Version;
import weka.estimators.Estimator;
import weka.estimators.NormalEstimator;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class WekaReference {

  static Instances load(String path) throws Exception {
    try (BufferedReader r = new BufferedReader(new FileReader(path))) {
      Instances data = new Instances(r);
      data.setClassIndex(data.numAttributes() - 1);
      return data;
    }
  }

  static String join(double[] v) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < v.length; i++) {
      if (i > 0) {
        sb.append(' ');
      }
      sb.append(Double.toString(v[i]));
    }
    return sb.toString();
  }

  static void posteriors(AbstractClassifier clf, Instances test) throws Exception {
    for (int i = 0; i < test.numInstances(); i++) {
      System.out.println("proba " + join(clf.distributionForInstance(test.instance(i))));
    }
  }

  public static void main(String[] args) throws Exception {
    System.out.println("weka_version " + Version.VERSION);
    String cmd = args[0];
    if (cmd.equals("nb")) {
      Instances train = load(args[1]);
      Instances test = load(args[2]);
      NaiveBayes nb = new NaiveBayes();
      nb.buildClassifier(train);
      System.out.println("options " + Utils.joinOptions(nb.getOptions()));
      Estimator[][] est = nb.getConditionalEstimators();
      for (int a = 0; a < est.length; a++) {
        for (int c = 0; c < est[a].length; c++) {
          if (est[a][c] instanceof NormalEstimator) {
            NormalEstimator ne = (NormalEstimator) est[a][c];
            System.out.println("normal " + a + " " + c + " " + ne.getMean() + " "
              + ne.getStdDev() + " " + ne.getPrecision());
          }
        }
      }
      posteriors(nb, test);
    } else if (cmd.equals("tan")) {
      Instances train = load(args[1]);
      Instances test = load(args[2]);
      double alpha = Double.parseDouble(args[3]);
      TAN search = new TAN();
      int score = -1;
      for (int t = 0; t < LocalScoreSearchAlgorithm.TAGS_SCORE_TYPE.length; t++) {
        if (LocalScoreSearchAlgorithm.TAGS_SCORE_TYPE[t].getReadable().equals(args[4])) {
          score = LocalScoreSearchAlgorithm.TAGS_SCORE_TYPE[t].getID();
        }
      }
      if (score < 0) {
        throw new IllegalArgumentException("Unknown score type " + args[4]);
      }
      search.setScoreType(new SelectedTag(score, LocalScoreSearchAlgorithm.TAGS_SCORE_TYPE));
      SimpleEstimator estimator = new SimpleEstimator();
      estimator.setAlpha(alpha);
      BayesNet bn = new BayesNet();
      bn.setSearchAlgorithm(search);
      bn.setEstimator(estimator);
      bn.buildClassifier(train);
      System.out.println("options " + Utils.joinOptions(bn.getOptions()));
      int classIndex = train.classIndex();
      for (int a = 0; a < train.numAttributes(); a++) {
        if (a == classIndex) {
          continue;
        }
        StringBuilder sb = new StringBuilder("parents " + a);
        for (int p = 0; p < bn.getParentSet(a).getNrOfParents(); p++) {
          int parent = bn.getParentSet(a).getParent(p);
          if (parent != classIndex) {
            sb.append(' ').append(parent);
          }
        }
        System.out.println(sb);
      }
      posteriors(bn, test);
    } else if (cmd.equals("aode")) {
      Instances train = load(args[1]);
      Instances test = load(args[2]);
      AODE aode = new AODE();
      aode.setOptions(new String[] {"-F", args[3]});
      aode.buildClassifier(train);
      System.out.println("options " + Utils.joinOptions(aode.getOptions()));
      posteriors(aode, test);
    } else if (cmd.equals("mdl")) {
      Instances data = load(args[1]);
      Discretize disc = new Discretize();
      disc.setInputFormat(data);
      Filter.useFilter(data, disc);
      System.out.println("options " + Utils.joinOptions(disc.getOptions()));
      for (int a = 0; a < data.numAttributes(); a++) {
        if (a == data.classIndex()) {
          continue;
        }
        double[] cuts = disc.getCutPoints(a);
        System.out.println("cuts " + a + (cuts == null ? "" : " " + join(cuts)));
      }
    } else {
      throw new IllegalArgumentException("Unknown command " + cmd);
    }
  }
}
