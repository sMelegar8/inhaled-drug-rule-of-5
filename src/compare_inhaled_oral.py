"""Step 5 - Compare inhaled and oral drugs, and test how robust the comparison is.

The headline comparison of the Rule of 5 case study: for each property, a
Mann-Whitney U test between the 46 lung-targeted inhaled drugs and the 1,108
oral drugs, with the rank-biserial correlation r as effect size and a Holm
correction across the properties.

A difference between the two routes can also be a difference between the
therapeutic classes that happen to be given by each route. The inhaled group is
made of four classes, and the "other inhaled" one holds drugs that are not
designed to stay in the lung (inhaled antibiotics, a diagnostic sugar, an
antiviral). This script therefore repeats the comparison on subsets:

  - respiratory classes only  : the 29 drugs whose ATC code says they treat
                                airway disease (beta2-agonists, antimuscarinics,
                                corticosteroids). The pre-specifiable subset.
  - leave one class out       : one scenario per class, to see which class
                                carries each result.
  - without also-oral drugs   : the 15 inhaled drugs that also have oral
                                products, already reported in the case study.
  - without extreme logP      : post-hoc diagnostic, flagged as such: the six
                                inhaled drugs with ALogP < -2.

A result that survives every subset is a property of the route. A result that
disappears when a handful of molecules leave is a property of those molecules.

Output: reports/inhaled_vs_oral.csv, reports/inhaled_vs_oral_sensitivity.csv,
reports/figures/sensitivity_by_class.png (copied to app/figures/), and the
numbers printed in the shape the case study page uses.
Usage:  py -3.13 src/compare_inhaled_oral.py      (after cleaning.py)
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_trends import respiratory_class  # noqa: E402  (same class rules as the time-trend study)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "pharma_inhalation.sqlite"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Properties compared, with the labels used on the case study page.
PROPERTIES = {
    "hbd": "H-bond donors",
    "qed_weighted": "QED drug-likeness",
    "aromatic_rings": "Aromatic rings",
    "psa": "Polar surface area (Å²)",
    "hba": "H-bond acceptors",
    "rtb": "Rotatable bonds",
    "alogp": "ALogP",
    "heavy_atoms": "Heavy atoms",
    "mw_freebase": "Molecular weight (Da)",
    "num_ro5_violations": "Rule-of-five violations",
}
RESPIRATORY = ("β2-agonists", "antimuscarinics", "corticosteroids")
EXTREME_LOGP = -2.0
ALPHA = 0.05


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the order given."""
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def compare(inhaled: pd.DataFrame, oral: pd.DataFrame, prop: str) -> dict:
    """Mann-Whitney U with the rank-biserial correlation as effect size."""
    a, b = inhaled[prop].dropna(), oral[prop].dropna()
    if len(a) < 3:
        return {"n_inhaled": len(a), "r": np.nan, "p": np.nan, "median_inhaled": np.nan, "median_oral": np.nan}
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"n_inhaled": len(a), "r": 2 * u.statistic / (len(a) * len(b)) - 1, "p": u.pvalue,
            "median_inhaled": a.median(), "median_oral": b.median()}


def scenarios(inhaled: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
    """Subsets of the inhaled group, with why each one is left out."""
    out = {"All inhaled drugs": (inhaled, "the published comparison")}
    out["Respiratory classes only"] = (
        inhaled[inhaled["cls"].isin(RESPIRATORY)],
        "only drugs whose ATC code says they treat airway disease")
    for cls in sorted(inhaled["cls"].unique()):
        out[f"Without {cls}"] = (inhaled[inhaled["cls"] != cls], f"drops the {cls}")
    out["Without also-oral drugs"] = (
        inhaled[inhaled["also_oral"] == 0], "drops the inhaled drugs that also have oral products")
    out[f"Without ALogP < {EXTREME_LOGP:g} (post-hoc)"] = (
        inhaled[~(inhaled["alogp"] < EXTREME_LOGP)], "post-hoc: drops the most polar inhaled drugs")
    return out


def figure(sens: pd.DataFrame, order: list[str]) -> None:
    """Effect size of each property across the subsets, so shrinking results are visible."""
    scenario_names = list(dict.fromkeys(sens["scenario"]))
    keep = [s for s in scenario_names if s in ("All inhaled drugs", "Respiratory classes only",
                                               "Without also-oral drugs", f"Without ALogP < {EXTREME_LOGP:g} (post-hoc)")]
    colors = {"All inhaled drugs": "#0b0b0b", "Respiratory classes only": "#2a78d6",
              "Without also-oral drugs": "#1baf7a", f"Without ALogP < {EXTREME_LOGP:g} (post-hoc)": "#eb6834"}
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e1e0d9", "axes.axisbelow": True}):
        fig, ax = plt.subplots(figsize=(10, 5.6))
        offsets = np.linspace(0.26, -0.26, len(keep))
        for scenario, off in zip(keep, offsets):
            sub = sens[sens["scenario"] == scenario].set_index("property").reindex(order)
            y = np.arange(len(order)) + off
            sig = sub["p"] < ALPHA
            ax.scatter(sub["r"][sig], y[sig.to_numpy()], s=58, color=colors[scenario], label=scenario, zorder=3)
            ax.scatter(sub["r"][~sig], y[(~sig).to_numpy()], s=58, facecolor="white",
                       edgecolor=colors[scenario], linewidth=1.6, zorder=3)
            n = int(sub["n_inhaled"].max())
            ax.plot([], [], " ", label=f"    n = {n} inhaled")
        ax.axvline(0, color="#898781", linewidth=1)
        ax.set_yticks(np.arange(len(order)), [PROPERTIES[p] for p in order])
        ax.invert_yaxis()
        ax.set_xlabel("effect size r (rank-biserial): right = higher in inhaled drugs")
        ax.set_title("How much of the inhaled–oral difference survives when subsets are removed")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.text(0.99, -0.14, "filled = significant (p < 0.05)   ·   hollow = not significant",
                transform=ax.transAxes, ha="right", fontsize=9, color="#52514e")
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "sensitivity_by_class.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    (ROOT / "app" / "figures").mkdir(exist_ok=True)
    shutil.copy(FIGURES / "sensitivity_by_class.png", ROOT / "app" / "figures" / "sensitivity_by_class.png")


def main() -> None:
    con = sqlite3.connect(DB)
    data = pd.read_sql_query("SELECT * FROM inhalation_dataset", con)
    con.close()
    inhaled = data[data.route_group == "inhaled"].copy()
    oral = data[data.route_group == "oral"]
    inhaled["cls"] = inhaled.apply(respiratory_class, axis=1)
    print(f"{len(inhaled)} inhaled vs {len(oral)} oral drugs")
    print("inhaled classes:", inhaled["cls"].value_counts().to_dict())

    # Headline comparison, Holm-corrected across the properties.
    main_rows = [{"property": p, "label": PROPERTIES[p], **compare(inhaled, oral, p)} for p in PROPERTIES]
    main_table = pd.DataFrame(main_rows)
    main_table["p_holm"] = holm(main_table["p"].to_numpy())
    main_table = main_table.sort_values("p_holm")
    main_table.to_csv(REPORTS / "inhaled_vs_oral.csv", index=False)
    print("\nHeadline comparison (Holm-corrected):")
    with pd.option_context("display.width", 200):
        print(main_table[["label", "median_inhaled", "median_oral", "r", "p", "p_holm"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # The same comparison on each subset.
    rows = []
    for name, (subset, why) in scenarios(inhaled).items():
        for prop in PROPERTIES:
            rows.append({"scenario": name, "why": why, "property": prop, "label": PROPERTIES[prop],
                         **compare(subset, oral, prop)})
    sens = pd.DataFrame(rows)
    sens.to_csv(REPORTS / "inhaled_vs_oral_sensitivity.csv", index=False)

    order = main_table["property"].tolist()
    wide = sens.pivot(index="property", columns="scenario", values="r").reindex(order)
    sig = sens.pivot(index="property", columns="scenario", values="p").reindex(order) < ALPHA
    print("\nEffect size r by subset (* = p < 0.05 within that subset, uncorrected):")
    shown = wide.copy()
    for col in wide.columns:
        shown[col] = [f"{v:+.2f}{'*' if s else ' '}" for v, s in zip(wide[col], sig[col])]
    shown.index = [PROPERTIES[p] for p in shown.index]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(shown.to_string())

    # Which results survive the principled subset, and which do not.
    survived, lost = [], []
    for prop in order:
        row = sens[(sens.scenario == "Respiratory classes only") & (sens.property == prop)].iloc[0]
        base = main_table[main_table.property == prop].iloc[0]
        if base["p_holm"] < ALPHA:
            (survived if row["p"] < ALPHA else lost).append(
                f"{PROPERTIES[prop]} ({base['r']:+.2f} -> {row['r']:+.2f}, p = {row['p']:.2f})")
    print("\nAmong the results significant in the published comparison:")
    print("  survive on respiratory classes only:", "; ".join(survived) or "none")
    print("  do NOT survive:", "; ".join(lost) or "none")
    drivers = inhaled[inhaled.alogp < EXTREME_LOGP].sort_values("alogp")
    print(f"\nThe {len(drivers)} inhaled drugs with ALogP < {EXTREME_LOGP:g} (post-hoc diagnostic):")
    print(drivers[["pref_name", "alogp", "psa", "hbd", "cls"]].to_string(index=False))

    figure(sens, order)
    print("\nSaved reports/inhaled_vs_oral.csv, reports/inhaled_vs_oral_sensitivity.csv and the figure.")

    # The rows the case study page hard-codes, so the page and the data cannot drift apart.
    print("\n--- EFFECTS array for app/case-study.html ---")
    for _, r in main_table.iterrows():
        med = lambda v: f"{v:g}" if abs(v) >= 0.01 else f"{v:.3g}"
        print(f'  ["{r.label}", {med(r.median_inhaled)}, {med(r.median_oral)}, {r.r:.3f}, {r.p_holm:.3g}],')


if __name__ == "__main__":
    main()
