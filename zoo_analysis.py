"""
Zoo analysis: do the baseline's mistakes (and near-mistakes) land on bears from the same zoo?

The 13 bears live in 6 zoos (data/PolarBearVidID/animal_db.csv): five zoos with 2 bears and
Hannover with 3. Zoo-mates share an enclosure, lighting and cameras, so if the model leans on
the scene, its mistakes should land on zoo-mates far more often than chance.

Chance: if the wrong bear were picked at random from the other 12, it would be a zoo-mate
with probability 1/12 (a bear in a 2-bear zoo) or 2/12 (a Hannover bear).

Everything uses the validation videos (queries) against the training videos (gallery).
The test split is never used.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

WHITE, INK, MUTED, GRID_LINE = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
BAR_COLOR = "#2a78d6"


def load_zoos(path="data/PolarBearVidID/animal_db.csv"):
    """Which zoo each bear lives in, keyed like '000'."""
    db = pd.read_csv(path)
    return pd.Series(db["zoo"].values, index=pd.Index([f"{i:03d}" for i in db["id"]], name="bear"), name="zoo")


def add_zoo_columns(results, zoos):
    """
    Adds to every validation video (rows from heatmaps.match_query_videos):
      zoo, zoo_mates:          the bear's zoo and how many other bears live there
      chance_zoo_mate:         chance that a random other bear is a zoo-mate (1/12 or 2/12)
      predicted_zoo:           zoo of the predicted bear
      predicted_zoo_mate:      WRONG, and the predicted bear lives in the same zoo
      closest_wrong:           the most similar bear that is not the true bear (the runner-up;
                               for a wrong video this is the predicted bear)
      closest_wrong_zoo_mate:  that bear lives in the same zoo
    """
    out = results.copy()
    zoo_size = zoos.map(zoos.value_counts())
    out["zoo"] = out["identity"].map(zoos)
    out["zoo_mates"] = out["identity"].map(zoo_size) - 1
    out["chance_zoo_mate"] = out["zoo_mates"] / (len(zoos) - 1)
    out["predicted_zoo"] = out["predicted"].map(zoos)
    out["predicted_zoo_mate"] = ~out["correct"] & (out["predicted_zoo"] == out["zoo"])
    out["closest_wrong"] = out["other_best_identity"]
    out["closest_wrong_zoo_mate"] = out["closest_wrong"].map(zoos) == out["zoo"]
    return out


def zoo_table(res, zoos):
    """One row per zoo: validation videos, correct, mistakes, and which zoo the mistakes went to."""
    wrong = ~res["correct"]
    g = res.assign(mistakes=wrong,
                   mistakes_to_same_zoo=res["predicted_zoo_mate"],
                   mistakes_to_other_zoo=wrong & ~res["predicted_zoo_mate"])
    table = g.groupby("zoo").agg(
        videos=("sequence", "size"),
        correct=("correct", "sum"),
        mistakes=("mistakes", "sum"),
        mistakes_to_same_zoo=("mistakes_to_same_zoo", "sum"),
        mistakes_to_other_zoo=("mistakes_to_other_zoo", "sum"),
        runner_up_same_zoo=("closest_wrong_zoo_mate", "sum"),
    )
    table = table.reindex(pd.unique(zoos.values))            # zoos in bear order
    table.insert(0, "bears", [", ".join(zoos.index[zoos == z]) for z in table.index])
    table.loc["All zoos"] = ["all 13"] + table.drop(columns="bears").sum().tolist()
    table["accuracy_%"] = (table["correct"] / table["videos"] * 100).round(1)
    return table


def bear_accuracy(res):
    """One row per bear (grouped by zoo): validation videos, correct, accuracy."""
    out = res.groupby(["zoo", "identity"], sort=False).agg(videos=("correct", "size"), correct=("correct", "sum"))
    out["accuracy_%"] = (out["correct"] / out["videos"] * 100).round(1)
    return out.sort_index(level="identity")


def observed_vs_chance(res):
    """How often the bear the model leaned toward is a zoo-mate, observed vs expected by chance."""
    wrong, right = res[~res["correct"]], res[res["correct"]]
    rows = [
        ("Mistakes: the predicted bear", wrong["predicted_zoo_mate"], wrong["chance_zoo_mate"]),
        ("Correct videos: the runner-up bear", right["closest_wrong_zoo_mate"], right["chance_zoo_mate"]),
        ("All videos: the runner-up bear", res["closest_wrong_zoo_mate"], res["chance_zoo_mate"]),
    ]
    return pd.DataFrame([{
        "which bear": name,
        "videos": len(observed),
        "zoo_mate": int(observed.sum()),
        "observed_%": observed.mean() * 100,
        "expected_by_chance": chance.sum(),
        "chance_%": chance.mean() * 100,
    } for name, observed, chance in rows])


def _style(ax):
    ax.set_facecolor(WHITE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)


def plot_zoo_mate_share(summary, out_path="results/zoo/zoo_mate_share.png"):
    """Bar chart: % of mistakes / runner-ups that are zoo-mates, with the chance level on each bar."""
    rows = summary[summary["which bear"] != "All videos: the runner-up bear"].reset_index(drop=True)
    labels = [f"{rows.loc[0, 'videos']} mistakes:\nthe predicted bear",
              f"{rows.loc[1, 'videos']} correct videos:\nthe runner-up bear"]

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=WHITE)
    _style(ax)
    ax.yaxis.grid(True, color=GRID_LINE, linewidth=0.8, zorder=0)
    ax.bar(range(len(rows)), rows["observed_%"], color=BAR_COLOR, width=0.5, zorder=2)
    for x, r in rows.iterrows():
        ax.text(x, r["observed_%"] + 2, f"{r['observed_%']:.0f}%  ({r['zoo_mate']} of {r['videos']})",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.plot([x - 0.32, x + 0.32], [r["chance_%"]] * 2, color=INK, linestyle=(0, (4, 3)), linewidth=1.4, zorder=3)
        ax.text(x + 0.35, r["chance_%"], f"chance {r['chance_%']:.0f}%", va="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_xlim(-0.6, len(rows) - 0.1)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% that live in the same zoo as the true bear", fontsize=10.5, color=INK)
    ax.set_title("The model's mistakes and runner-ups are mostly zoo-mates", fontsize=13.5,
                 fontweight="bold", color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, "Validation videos vs training videos. Chance = a random other bear (1 or 2 zoo-mates out of 12).",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=WHITE)
    return fig, out_path


def plot_zoo_accuracy(table, out_path="results/zoo/zoo_accuracy.png"):
    """Horizontal bars: validation Rank-1 accuracy per zoo, plus all zoos together (gray) at the bottom."""
    labels = [f"{zoo}\n(bears {bears})" if zoo != "All zoos" else "All zoos\n(13 bears)"
              for zoo, bears in zip(table.index, table["bears"])]
    colors = [BAR_COLOR if zoo != "All zoos" else "#8a8984" for zoo in table.index]
    y = np.arange(len(table))[::-1].astype(float)           # first zoo on top, "All zoos" last
    y[-1] -= 0.4                                            # small gap before the total

    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.xaxis.grid(True, color=GRID_LINE, linewidth=0.8, zorder=0)
    ax.barh(y, table["accuracy_%"], color=colors, height=0.55, zorder=2)
    for yi, (_, r) in zip(y, table.iterrows()):
        ax.text(r["accuracy_%"] + 1, yi, f"{r['accuracy_%']:.1f}%  ({int(r['correct'])} of {int(r['videos'])})",
                va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, 118)
    ax.set_xticks(range(0, 101, 20))
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.tick_params(axis="x", labelsize=10, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.set_xlabel("Validation Rank-1 accuracy (%)", fontsize=10.5, color=INK)
    ax.set_title("Accuracy per zoo", fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, "Validation videos vs training videos (baseline: 20 frames averaged, cosine nearest neighbour).",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=WHITE)
    return fig, out_path


def _confusion_counts(true_bears, other_bears, bears):
    counts = pd.crosstab(pd.Categorical(true_bears, bears), pd.Categorical(other_bears, bears), dropna=False)
    return counts.reindex(index=bears, columns=bears, fill_value=0).to_numpy()


def plot_zoo_confusion(res, zoos, out_path="results/zoo/zoo_confusion.png"):
    """
    Two matrices (rows = true bear, columns = the other bear), with each zoo outlined:
      left:  the 24 mistakes (column = predicted bear)
      right: all validation videos (column = the most similar wrong bear, the runner-up)
    Counts inside the outlined squares are zoo-mates.
    """
    bears = list(zoos.index)
    wrong = res[~res["correct"]]
    panels = [(_confusion_counts(wrong["identity"], wrong["predicted"], bears),
               f"Mistakes ({len(wrong)}): true bear -> predicted bear"),
              (_confusion_counts(res["identity"], res["closest_wrong"], bears),
               f"All {len(res)} videos: true bear -> runner-up bear")]
    blocks, start = [], 0
    for zoo in pd.unique(zoos.values):
        size = int((zoos == zoo).sum())
        blocks.append((start, size, zoo))
        start += size

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), facecolor=WHITE)
    for ax, (counts, title) in zip(axes, panels):
        shown = np.ma.masked_where(np.eye(len(bears), dtype=bool), counts)
        ax.imshow(shown, cmap="Blues", vmin=0, vmax=max(1, counts.max()))
        for i in range(len(bears)):
            for j in range(len(bears)):
                if counts[i, j] and i != j:
                    dark = counts[i, j] > counts.max() * 0.55
                    ax.text(j, i, int(counts[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if dark else INK)
        for s, size, zoo in blocks:
            ax.add_patch(Rectangle((s - 0.5, s - 0.5), size, size, fill=False, edgecolor="#eb6834", linewidth=2))
        ax.set_xticks(range(len(bears)))
        ax.set_xticklabels(bears, fontsize=8.5, color=INK, rotation=90)
        ax.set_yticks(range(len(bears)))
        ax.set_yticklabels([f"{b}  {zoos[b]}" for b in bears], fontsize=8.5, color=INK)
        ax.set_xlabel("other bear", fontsize=10, color=INK)
        ax.set_ylabel("true bear", fontsize=10, color=INK)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_color(GRID_LINE)
    fig.suptitle("Who is mistaken for whom? Orange squares = bears from the same zoo",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=WHITE)
    return fig, out_path


def load_names(path="data/PolarBearVidID/animal_db.csv"):
    """Each bear's name, keyed like '000'."""
    db = pd.read_csv(path)
    return pd.Series(db["name"].values, index=pd.Index([f"{i:03d}" for i in db["id"]], name="bear"), name="name")
