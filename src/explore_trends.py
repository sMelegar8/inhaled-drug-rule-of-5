"""Exploration - have inhaled drugs become more lipophilic over time?

Finding tested here (exploratory, on the cleaned analysis set):
  among FDA inhaled drugs, the year of first approval correlates with
  lipophilicity (alogp), size and flexibility (rotatable bonds), within each
  respiratory class, and the lipophilicity trend is steeper than for oral drugs.

Checks: Spearman correlations with bootstrap 95% CIs, a class-adjusted linear
model (property ~ year + class) with a permutation test that shuffles years
within class, and the inhaled - oral difference in correlation.

Output: printed results, reports/exploration_trends.csv, reports/figures/trend_logp_year.png
Usage:  py -3.13 src/explore_trends.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
REPORTS = ROOT / "reports"
SEED = 42
PROPERTIES = ["alogp", "rtb", "mw_freebase", "psa", "hbd", "aromatic_rings"]
# Respiratory molecules without an ATC code in ChEMBL, assigned by pharmacological class.
NO_ATC_CLASS = {"ARFORMOTEROL": "β2-agonists", "LEVOSALBUTAMOL": "β2-agonists", "VILANTEROL": "β2-agonists",
                "TRIAMCINOLONE ACETONIDE": "corticosteroids"}
CLASS_COLORS = {"β2-agonists": "#2a78d6", "antimuscarinics": "#eb6834", "corticosteroids": "#1baf7a",
                "other inhaled": "#6250d6"}


def respiratory_class(row: pd.Series) -> str:
    atc = row["atc_codes"]
    if any(code in atc for code in ("R03AA", "R03AB", "R03AC")):
        return "β2-agonists"
    if "R03BB" in atc:
        return "antimuscarinics"
    if "R03BA" in atc:
        return "corticosteroids"
    return NO_ATC_CLASS.get(row["pref_name"], "other inhaled")


def bootstrap_rho(x: np.ndarray, y: np.ndarray, rng, n_boot: int) -> np.ndarray:
    n = len(x)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        out[b] = stats.spearmanr(x[i], y[i]).statistic
    return out


def class_adjusted_slope(df: pd.DataFrame, prop: str, rng, n_perm: int) -> tuple[float, float]:
    """Slope of prop on year with one intercept per class; permutation p-value."""
    dummies = pd.get_dummies(df["cls"]).astype(float).to_numpy()
    y = df[prop].to_numpy()

    def slope(years: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(np.column_stack([dummies, years]), y, rcond=None)
        return coef[-1]

    observed = slope(df["first_approval"].to_numpy())
    perms = np.array([slope(df.groupby("cls")["first_approval"]
                            .transform(lambda s: rng.permutation(s.to_numpy())).to_numpy())
                      for _ in range(n_perm)])
    return observed, (np.sum(np.abs(perms) >= abs(observed)) + 1) / (n_perm + 1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect(DB)
    data = pd.read_sql_query("SELECT * FROM inhalation_dataset WHERE first_approval IS NOT NULL", con)
    inhaled = data[data.route_group == "inhaled"].copy()
    oral = data[data.route_group == "oral"]
    inhaled["cls"] = inhaled.apply(respiratory_class, axis=1)
    respiratory = inhaled[inhaled.cls != "other inhaled"]
    print(f"Inhaled {len(inhaled)} (respiratory classes {len(respiratory)}), oral {len(oral)}")

    rows = []
    for prop in PROPERTIES:
        rho_i = stats.spearmanr(inhaled.first_approval, inhaled[prop])
        rho_o = stats.spearmanr(oral.first_approval, oral[prop])
        boot_i = bootstrap_rho(inhaled.first_approval.to_numpy(), inhaled[prop].to_numpy(), rng, 4000)
        boot_o = bootstrap_rho(oral.first_approval.to_numpy(), oral[prop].to_numpy(), rng, 4000)
        slope, p_perm = class_adjusted_slope(respiratory, prop, rng, 5000)
        rows.append({
            "property": prop,
            "rho_inhaled": rho_i.statistic, "ci_inhaled": np.percentile(boot_i, [2.5, 97.5]).round(2).tolist(),
            "rho_oral": rho_o.statistic, "ci_oral": np.percentile(boot_o, [2.5, 97.5]).round(2).tolist(),
            "ci_difference": np.percentile(boot_i - boot_o, [2.5, 97.5]).round(2).tolist(),
            "class_adjusted_per_decade": slope * 10, "p_permutation": p_perm,
        })
    result = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(result.to_string(index=False))
    for cls, g in respiratory.groupby("cls"):
        print(f"  {cls}: n={len(g)}, rho(year, alogp) = {stats.spearmanr(g.first_approval, g.alogp).statistic:+.2f}")
    REPORTS.mkdir(exist_ok=True)
    result.to_csv(REPORTS / "exploration_trends.csv", index=False)

    # Figure: logP by year of first approval
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(oral.first_approval, oral.alogp, s=12, color="#c3c2b7", alpha=0.6, label=f"oral (n={len(oral)})", zorder=1)
    for cls, color in CLASS_COLORS.items():
        g = inhaled[inhaled.cls == cls]
        ax.scatter(g.first_approval, g.alogp, s=55, color=color, edgecolor="white", linewidth=0.8,
                   label=f"{cls if cls.startswith('other') else 'inhaled: ' + cls} (n={len(g)})", zorder=3)
    for df, color, label in [(oral, "#898781", "oral trend"), (inhaled, "#0b0b0b", "inhaled trend")]:
        slope, intercept = np.polyfit(df.first_approval, df.alogp, 1)
        xs = np.array([df.first_approval.min(), df.first_approval.max()])
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=2, linestyle="--" if df is oral else "-",
                label=f"{label}: {slope * 10:+.2f} logP/decade", zorder=2)
    offsets = {"ALBUTEROL": (6, -12), "SALMETEROL": (6, 4), "IPRATROPIUM": (-20, 9), "TIOTROPIUM": (6, -12),
               "BUDESONIDE": (6, -12), "UMECLIDINIUM": (-30, 10), "VILANTEROL": (6, -13)}
    for name, offset in offsets.items():
        r = inhaled[inhaled.pref_name == name].iloc[0]
        ax.annotate(name.title(), (r.first_approval, r.alogp), xytext=offset, textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("Year of first approval")
    ax.set_ylabel("Calculated logP (ALogP)")
    ax.set_title("Inhaled drugs became more lipophilic faster than oral drugs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "trend_logp_year.png", dpi=150)
    print("Figure: reports/figures/trend_logp_year.png")


if __name__ == "__main__":
    main()
