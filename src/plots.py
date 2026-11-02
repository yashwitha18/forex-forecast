"""Figures for the report and slides.
01-02: Member A (data).  03-07: Member B (results)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#2b2b2b", "#6b6a63", "#e6e5df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _style(ax, title=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _save(fig, path: Path, rect=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path.name}")


def rates(fx: dict, bounds: dict, colors: dict, path: Path):
    """Paper Figure 1, as small multiples (the pairs have different scales)."""
    n = len(fx)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    for ax, (k, df) in zip(np.atleast_1d(axes), fx.items()):
        ax.plot(df["date"], df["close"], color=colors.get(k, INK), linewidth=1)
        v0, t0 = bounds[k]
        ax.axvspan(v0, t0, color=GRID, alpha=0.5, lw=0)
        ax.axvspan(t0, df["date"].max(), color="#cde2fb", alpha=0.6, lw=0)
        _style(ax, f"{k} per 1 USD", k)
    ax.text(0.99, 0.95, "grey = validation, blue = test", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=MUTED)
    _save(fig, path)


def returns_hist(fx: dict, colors: dict, path: Path):
    fig, axes = plt.subplots(1, len(fx), figsize=(4 * len(fx), 3.3), sharey=True)
    for ax, (k, df) in zip(np.atleast_1d(axes), fx.items()):
        r = np.log(df["close"]).diff().dropna() * 100
        ax.hist(r, bins=80, color=colors.get(k, INK), edgecolor="white", linewidth=0.3)
        _style(ax, f"{k}: daily log return (%)")
    _save(fig, path)


def grouped_dots(table: pd.DataFrame, value: str, group: str, title: str,
                 path: Path, ref: float | None = None, ref_label: str = ""):
    """Dot plot: one row per (model, group). Dots instead of bars because the
    interesting differences are tiny (e.g. 0.49 vs 0.52 accuracy) and bars
    must start at zero, which would hide them."""
    order = list(dict.fromkeys(table[group]))
    piv = table.pivot(index="model", columns=group, values=value)[order]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    n = len(order)
    x = np.arange(len(piv))
    if ref is not None:
        ax.axhline(ref, color=INK, linewidth=1, linestyle="--")
        ax.text(len(piv) - 0.5, ref, ref_label, ha="right", va="bottom", fontsize=8, color=INK)
    for i, col in enumerate(order):
        ax.scatter(x + (i - (n - 1) / 2) * 0.18, piv[col], s=55, color=SERIES[i % len(SERIES)],
                   edgecolors="white", linewidths=1.5, label=col, zorder=3)
    ax.set_xticks(x, piv.index, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    _style(ax, title)
    _save(fig, path)


def pred_vs_actual(frames: dict, model: str, path: Path):
    """Paper Figure 2: actual vs predicted log returns over the test period."""
    n = len(frames)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    for ax, (k, f) in zip(np.atleast_1d(axes), frames.items()):
        ax.scatter(f["next_date"], f["next_ret"] * 100, s=5, color=INK, alpha=0.5, label="actual")
        ax.scatter(f["next_date"], f["pred_ret"] * 100, s=5, color=SERIES[0], label="predicted")
        ax.axhline(0, color=MUTED, linewidth=0.6)
        _style(ax, f"{k}: next-day log return (%) — {model}")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, markerscale=3, loc="upper right")
    _save(fig, path)


def cash_paths(paths: dict, model: str, combos: list, cost: float, path: Path):
    """Paper Figure 3, plus benchmarks: hold USD (flat $100) and momentum."""
    fig, axes = plt.subplots(len(combos), 1, figsize=(10, 2.6 * len(combos)), sharex=True)
    for ax, combo in zip(np.atleast_1d(axes), combos):
        for (m, c, k), color, lab in [
            ((model, combo, 0.0), SERIES[0], f"{model}, no cost (paper)"),
            ((model, combo, cost), SERIES[1], f"{model}, {cost * 1e4:.0f} bp/side"),
            (("Momentum", combo, 0.0), MUTED, "Momentum benchmark, no cost"),
        ]:
            p = paths.get((m, c, k))
            if p is not None:
                ax.plot(p["date"], p["cash"], color=color, linewidth=1.2, label=lab)
        ax.axhline(100, color=INK, linewidth=0.8, linestyle="--")
        _style(ax, f"Cash ($), trading {combo}")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=3, loc="lower center")
    _save(fig, path, rect=(0, 0.03, 1, 1))


def importances(imp: pd.Series, title: str, path: Path, top: int = 15):
    imp = imp.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    _style(ax, title)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    _save(fig, path)
