// Reads {columns: {name: values[]}, groups: labels[], classes: labels[]} as JSON on stdin,
// runs app/stats.js and prints the results as JSON (used by validate_stats.py).
const path = require("path");
const Stats = require(path.join(__dirname, "..", "app", "stats.js"));
// [degrees of freedom, statistic] pairs for the chi-square tail used by Kruskal-Wallis
const CHI2_CASES = [[1, 0.5], [1, 3.84], [2, 5.99], [4, 9.49], [4, 2], [10, 25], [0.5, 0.1], [20, 60]];

let input = "";
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  const { columns, groups, classes } = JSON.parse(input);
  const names = Object.keys(columns);
  const cols = names.map((n) => columns[n]);
  const pick = (values, labels, label) =>
    values.filter((v, i) => labels[i] === label && Stats.isNum(v));

  const describe = Object.fromEntries(names.map((n, i) => [n, Stats.describe(cols[i])]));
  const pearson = Stats.correlationMatrix(cols, "pearson").r;
  const spearman = Stats.correlationMatrix(cols, "spearman").r;
  const mw = names.map((n, i) => Stats.mannWhitney(pick(cols[i], groups, "inhaled"), pick(cols[i], groups, "oral")));
  const holm = Stats.holm(mw.map((r) => r.p));
  const classesLabels = [...new Set(classes.filter((s) => s))].sort();
  const psa = cols[names.indexOf("psa")];
  const kw = Stats.kruskal(classesLabels.map((s) => pick(psa, classes, s)));

  const complete = cols[0].map((_, r) => cols.map((c) => c[r])).filter((row) => row.every(Stats.isNum));
  const pca = Stats.pca(complete, { standardize: true });
  const pcaRaw = Stats.pca(complete, { standardize: false });

  process.stdout.write(JSON.stringify({
    names, describe, pearson, spearman, mw, holm, classesLabels, kw,
    pca: { explained: pca.explained, eigenvalues: pca.eigenvalues, coefficients: pca.coefficients,
           loadings: pca.loadings, scores: pca.scores, n: pca.n },
    pcaRaw: { explained: pcaRaw.explained, loadings: pcaRaw.loadings },
    chi2Sf: CHI2_CASES.map(([df, x]) => Stats.gammaQ(df / 2, x / 2)),
  }));
});
