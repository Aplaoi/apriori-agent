"""Matplotlib visualisations for frequent itemsets and association rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((font for font in preferred if font in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _shorten(text: str, limit: int = 46) -> str:
    text = str(text).replace("⇒", "→")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _empty_chart(path: Path, title: str, message: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.text(0.5, 0.45, message, ha="center", va="center", fontsize=12, color="#64748b")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_frequent_itemsets(
    itemsets: list[dict[str, Any]],
    path: Path,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of the highest-support frequent itemsets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not itemsets:
        _empty_chart(path, "频繁项集 · Frequent Itemsets", "没有找到满足条件的频繁项集")
        return path

    plt = _setup_matplotlib()
    selected = sorted(itemsets, key=lambda entry: entry["support"], reverse=True)[
        :top_n
    ]
    selected = list(reversed(selected))

    labels = [
        _shorten(entry.get("display") or " + ".join(entry.get("items", [])))
        for entry in selected
    ]
    supports = [float(entry["support"]) * 100 for entry in selected]
    sizes = [int(entry.get("size", len(entry.get("items", [])))) for entry in selected]
    colors = plt.cm.viridis(
        [
            (size - min(sizes)) / max(max(sizes) - min(sizes), 1) * 0.72 + 0.12
            for size in sizes
        ]
    )

    fig, ax = plt.subplots(figsize=(10.5, max(4.4, 0.44 * len(selected) + 1.6)))
    bars = ax.barh(labels, supports, color=colors, edgecolor="white", linewidth=0.8)
    max_support = max(supports) if supports else 1
    for bar, support in zip(bars, supports):
        ax.text(
            bar.get_width() + max_support * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{support:.1f}%",
            va="center",
            fontsize=9,
            color="#334155",
        )
    ax.set_xlabel("支持度 Support (%)", fontsize=10)
    ax.set_title(
        f"频繁项集 Top {len(selected)} · Frequent Itemsets",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.set_xlim(0, max_support * 1.18)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_rules(rules: list[dict[str, Any]], path: Path, top_n: int = 12) -> Path:
    """Horizontal bar chart of the strongest association rules by confidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rules:
        _empty_chart(path, "关联规则 · Association Rules", "当前参数下没有找到关联规则")
        return path

    plt = _setup_matplotlib()
    selected = sorted(rules, key=lambda entry: entry["confidence"], reverse=True)[
        :top_n
    ]
    selected = list(reversed(selected))

    labels = [_shorten(entry.get("display", ""), 56) for entry in selected]
    confidences = [float(entry["confidence"]) * 100 for entry in selected]
    lifts = [float(entry.get("lift", 0)) for entry in selected]

    fig, ax = plt.subplots(figsize=(10.5, max(4.4, 0.5 * len(selected) + 1.6)))
    bars = ax.barh(
        labels,
        confidences,
        color="#2563eb",
        alpha=0.86,
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, confidence, lift in zip(bars, confidences, lifts):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{confidence:.1f}%  (lift {lift:.2f})",
            va="center",
            fontsize=8.6,
            color="#334155",
        )
    ax.set_xlabel("置信度 Confidence (%)", fontsize=10)
    ax.set_title(
        f"强关联规则 Top {len(selected)} · Association Rules",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.set_xlim(0, 118)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_charts(result: dict[str, Any], job_dir: Path) -> dict[str, str]:
    itemset_path = plot_frequent_itemsets(result["itemsets"], job_dir / "itemsets.svg")
    rules_path = plot_rules(result["rules"], job_dir / "rules.svg")
    return {"itemsets": str(itemset_path), "rules": str(rules_path)}
