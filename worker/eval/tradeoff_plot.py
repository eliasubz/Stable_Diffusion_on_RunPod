"""Identity vs. prompt-adherence plot from a run's summary.csv.

    python eval/tradeoff_plot.py experiments/<run>/summary.csv [out.png] [--title "..."]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {"text_only": "#888888", "ip_adapter": "#d62728", "lora": "#1f77b4"}


def spread(ys, min_gap):
    """Nudge sorted label positions apart so neighbors are at least min_gap apart."""
    out = []
    for y in ys:
        out.append(max(y, out[-1] + min_gap) if out else y)
    return out


def plot_tradeoff(summary: pd.DataFrame, path: str, title: str):
    s = summary.reset_index() if "method" not in summary.columns else summary
    fig, ax = plt.subplots(figsize=(9, 6))

    for method, color in STYLE.items():
        g = s[s.method == method].sort_values("param")
        if g.empty:
            continue
        sem = lambda c: g[f"{c}_std"] / g["n"] ** 0.5  # noqa: E731
        ax.errorbar(g.clip_t_mean, g.dino_crop_mean, xerr=sem("clip_t"), yerr=sem("dino_crop"),
                    fmt="-o", color=color, label=method.replace("_", " "), capsize=3, zorder=3)

    # IP-Adapter points are far apart: label them in place, on the left (the LoRA cluster is to the right)
    for _, r in s[s.method == "ip_adapter"].iterrows():
        ax.annotate(f"scale {r.param:g}", (r.clip_t_mean, r.dino_crop_mean), textcoords="offset points",
                    xytext=(-8, 6), ha="right", fontsize=9, color=STYLE["ip_adapter"])

    # LoRA points cluster tightly: put labels in a column to the right, spread vertically, with leader lines
    # Labels are ordered by step count so they read top-down like the curve
    lora = s[s.method == "lora"].sort_values("param")
    if not lora.empty:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        right_edge = (lora.clip_t_mean + lora.clip_t_std / lora.n ** 0.5).max()  # clear the error bars
        label_x = right_edge + 0.04 * (x1 - x0)
        ys = spread(lora.dino_crop_mean.tolist(), min_gap=0.045 * (y1 - y0))
        for (_, r), ly in zip(lora.iterrows(), ys):
            ax.annotate(f"{int(r.param)} steps · {r.train_min:.1f} min", xy=(r.clip_t_mean, r.dino_crop_mean),
                        xytext=(label_x, ly), textcoords="data", fontsize=9, va="center", color=STYLE["lora"],
                        arrowprops=dict(arrowstyle="-", color=STYLE["lora"], lw=0.6, alpha=0.6))
        ax.set_xlim(x0, label_x + 0.30 * (x1 - x0))
        ax.set_ylim(y0, max(y1, ys[-1] + 0.03 * (y1 - y0)))

    ax.set_xlabel("prompt adherence (CLIP-T) →")
    ax.set_ylabel("identity on held-out photos (DINO, pet crop) →")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("summary_csv")
    p.add_argument("out", nargs="?")
    p.add_argument("--title", default="LoRA training steps vs. IP-Adapter scale")
    a = p.parse_args()
    plot_tradeoff(pd.read_csv(a.summary_csv), a.out or a.summary_csv.replace("summary.csv", "tradeoff.png"), a.title)
