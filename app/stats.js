/* Statistics for the Inhalation SQL Workbench.
 *
 * Plain JavaScript, no dependencies, so it runs in the browser (window.Stats)
 * and in Node for validation against numpy/scipy (tests/validate_stats.py).
 * Conventions follow numpy/scipy defaults unless stated:
 *   - sample standard deviation (n - 1)
 *   - quantiles with linear interpolation (numpy "linear", R type 7)
 *   - Mann-Whitney U: asymptotic normal p-value with tie and continuity
 *     correction (scipy.stats.mannwhitneyu, method="asymptotic")
 *   - Kruskal-Wallis H with tie correction (scipy.stats.kruskal)
 *   - PCA on the correlation matrix when standardised (z-scores), on the
 *     covariance matrix otherwise; each component's sign is fixed so that its
 *     largest-magnitude coefficient is positive.
 */
(function (root) {
  "use strict";

  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const mean = (a) => sum(a) / a.length;

  function variance(a) {
    if (a.length < 2) return NaN;
    const m = mean(a);
    return sum(a.map((v) => (v - m) ** 2)) / (a.length - 1);
  }

  function quantileSorted(s, q) {
    if (!s.length) return NaN;
    const h = (s.length - 1) * q, lo = Math.floor(h), hi = Math.ceil(h);
    return s[lo] + (h - lo) * (s[hi] - s[lo]);
  }

  function describe(values) {
    const v = values.filter(isNum);
    const s = [...v].sort((a, b) => a - b);
    return {
      n: v.length, missing: values.length - v.length,
      mean: v.length ? mean(v) : NaN, sd: Math.sqrt(variance(v)),
      min: s[0] ?? NaN, q1: quantileSorted(s, 0.25), median: quantileSorted(s, 0.5),
      q3: quantileSorted(s, 0.75), max: s[s.length - 1] ?? NaN,
    };
  }

  /* Average ranks (1-based) and the tie term sum(t^3 - t). */
  function rank(values) {
    const idx = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const ranks = new Array(values.length);
    let tie = 0;
    for (let i = 0; i < idx.length;) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const r = (i + j) / 2 + 1, t = j - i + 1;
      for (let k = i; k <= j; k++) ranks[idx[k][1]] = r;
      tie += t ** 3 - t;
      i = j + 1;
    }
    return { ranks, tie };
  }

  function pearson(x, y) {
    const mx = mean(x), my = mean(y);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < x.length; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    return sxy / Math.sqrt(sxx * syy);
  }

  /* Pairwise-complete correlation matrix; method "pearson" or "spearman". */
  function correlationMatrix(columns, method = "pearson") {
    const p = columns.length;
    const r = Array.from({ length: p }, () => new Array(p).fill(NaN));
    const n = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = i; j < p; j++) {
        let x = [], y = [];
        columns[i].forEach((v, k) => { if (isNum(v) && isNum(columns[j][k])) { x.push(v); y.push(columns[j][k]); } });
        if (method === "spearman") { x = rank(x).ranks; y = rank(y).ranks; }
        const value = x.length > 2 ? (i === j ? 1 : pearson(x, y)) : NaN;
        r[i][j] = r[j][i] = value;
        n[i][j] = n[j][i] = x.length;
      }
    }
    return { r, n };
  }

  /* erfc with fractional error < 1.2e-7 everywhere (Numerical Recipes erfcc). */
  function erfc(x) {
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
      t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  const normalSf = (z) => 0.5 * erfc(z / Math.SQRT2);

  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5, ser = 1.000000000190015;
    tmp -= (x + 0.5) * Math.log(tmp);
    for (const cj of c) ser += cj / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }

  /* Regularised upper incomplete gamma Q(a, x). */
  function gammaQ(a, x) {
    if (x <= 0) return 1;
    const gln = lgamma(a);
    if (x < a + 1) {
      let ap = a, del = 1 / a, s = del;
      for (let n = 0; n < 1000; n++) { del *= x / ++ap; s += del; if (Math.abs(del) < Math.abs(s) * 1e-15) break; }
      return 1 - s * Math.exp(-x + a * Math.log(x) - gln);
    }
    const FPMIN = 1e-300;
    let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
    for (let i = 1; i < 1000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return Math.exp(-x + a * Math.log(x) - gln) * h;
  }

  /* Two-sided Mann-Whitney U for samples a and b. U is reported for a;
   * rankBiserial > 0 means values in a tend to be larger than in b. */
  function mannWhitney(a, b) {
    const n1 = a.length, n2 = b.length, n = n1 + n2;
    const { ranks, tie } = rank(a.concat(b));
    const u1 = sum(ranks.slice(0, n1)) - n1 * (n1 + 1) / 2;
    const mu = n1 * n2 / 2;
    const sigma = Math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))));
    const uMax = Math.max(u1, n1 * n2 - u1);
    const z = sigma > 0 ? (uMax - mu - 0.5) / sigma : 0;
    return { U: u1, z, p: Math.min(1, 2 * normalSf(z)), rankBiserial: 2 * u1 / (n1 * n2) - 1 };
  }

  /* Kruskal-Wallis H test for k >= 2 groups; epsilon2 = H / (n - 1). */
  function kruskal(groups) {
    const all = groups.flat(), n = all.length, k = groups.length;
    const { ranks, tie } = rank(all);
    let offset = 0, h = 0;
    for (const g of groups) {
      const r = sum(ranks.slice(offset, offset + g.length));
      h += r * r / g.length;
      offset += g.length;
    }
    h = 12 / (n * (n + 1)) * h - 3 * (n + 1);
    h /= 1 - tie / (n ** 3 - n);
    return { H: h, df: k - 1, p: gammaQ((k - 1) / 2, h / 2), epsilon2: h / (n - 1) };
  }

  /* Holm step-down adjustment of p-values (same order as input). */
  function holm(ps) {
    const order = ps.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
    const adj = new Array(ps.length);
    let running = 0;
    order.forEach(([p, i], rankIdx) => {
      running = Math.max(running, Math.min(1, (ps.length - rankIdx) * p));
      adj[i] = running;
    });
    return adj;
  }

  /* Eigen-decomposition of a symmetric matrix (cyclic Jacobi rotations).
   * Returns eigenvalues in descending order and eigenvectors as columns. */
  function eigenSymmetric(matrix) {
    const p = matrix.length;
    const a = matrix.map((row) => [...row]);
    const v = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += a[i][j] ** 2;
      if (off < 1e-22) break;
      for (let i = 0; i < p; i++) {
        for (let j = i + 1; j < p; j++) {
          if (Math.abs(a[i][j]) < 1e-300) continue;
          const theta = (a[j][j] - a[i][i]) / (2 * a[i][j]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < p; k++) {
            const aki = a[k][i], akj = a[k][j];
            a[k][i] = c * aki - s * akj; a[k][j] = s * aki + c * akj;
          }
          for (let k = 0; k < p; k++) {
            const aik = a[i][k], ajk = a[j][k];
            a[i][k] = c * aik - s * ajk; a[j][k] = s * aik + c * ajk;
          }
          for (let k = 0; k < p; k++) {
            const vki = v[k][i], vkj = v[k][j];
            v[k][i] = c * vki - s * vkj; v[k][j] = s * vki + c * vkj;
          }
        }
      }
    }
    const order = a.map((row, i) => [row[i], i]).sort((x, y) => y[0] - x[0]);
    return {
      values: order.map(([val]) => val),
      vectors: v.map((row) => order.map(([, i]) => row[i])),
    };
  }

  /* PCA on complete rows (n x p). Returns explained variance, coefficients
   * (eigenvectors, p x p), loadings (correlation of each variable with each
   * component when standardised, p x p) and scores (n x p). */
  function pca(rows, { standardize = true } = {}) {
    if (rows.length < 3 || rows[0].length < 2) throw new Error("PCA needs at least 3 complete rows and 2 variables.");
    const n = rows.length, p = rows[0].length;
    const means = [], sds = [];
    for (let j = 0; j < p; j++) {
      const col = rows.map((r) => r[j]);
      means.push(mean(col)); sds.push(Math.sqrt(variance(col)));
    }
    const z = rows.map((r) => r.map((v, j) => (v - means[j]) / (standardize ? sds[j] : 1)));
    const cov = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => {
      let s = 0; for (let k = 0; k < n; k++) s += z[k][i] * z[k][j];
      return s / (n - 1);
    }));
    const { values, vectors } = eigenSymmetric(cov);
    for (let c = 0; c < p; c++) {
      let best = 0;
      for (let j = 0; j < p; j++) if (Math.abs(vectors[j][c]) > Math.abs(vectors[best][c])) best = j;
      if (vectors[best][c] < 0) for (let j = 0; j < p; j++) vectors[j][c] = -vectors[j][c];
    }
    const eig = values.map((x) => Math.max(x, 0));
    const total = sum(eig);
    const scores = z.map((r) => vectors[0].map((_, c) => r.reduce((s, val, j) => s + val * vectors[j][c], 0)));
    const loadings = vectors.map((row, j) => row.map((coef, c) =>
      coef * Math.sqrt(eig[c]) / (standardize ? 1 : sds[j])));
    return {
      n, p, means, sds, eigenvalues: eig, explained: eig.map((x) => x / total),
      coefficients: vectors, loadings, scores,
    };
  }

  const Stats = { isNum, describe, quantileSorted, rank, pearson, correlationMatrix, erfc, normalSf,
    lgamma, gammaQ, mannWhitney, kruskal, holm, eigenSymmetric, pca };
  root.Stats = Stats;
  if (typeof module !== "undefined" && module.exports) module.exports = Stats;
})(typeof window !== "undefined" ? window : globalThis);
