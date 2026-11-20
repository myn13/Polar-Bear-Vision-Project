"""
Layer-wise analysis: how much individual polar-bear identity each stage of the frozen
MegaDescriptor-S-224 (Swin-Small) carries.

Nothing is trained. Each frame goes through the frozen model once, and we keep the output
of every stage, averaged over its spatial grid:
    stage1: 96 numbers   (56 x 56 grid, edges and colors)
    stage2: 192 numbers  (28 x 28 grid, textures)
    stage3: 384 numbers  (14 x 14 grid, parts)
    stage4: 768 numbers  (7 x 7 grid, before the final normalization)
    final:  768 numbers  (stage4 after the final normalization = the embedding the baseline uses)
Then the exact baseline is run with each stage's features: average a video's 20 frames,
match each validation video to its most similar training video (the gallery), and measure
Rank-1 accuracy. The training videos are needed only as the gallery for each stage.

The test split is never run through the model here.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

STAGES = ["stage1", "stage2", "stage3", "stage4", "final"]
STAGE_LABELS = {
    "stage1": "Stage 1\n96-d, 56x56",
    "stage2": "Stage 2\n192-d, 28x28",
    "stage3": "Stage 3\n384-d, 14x14",
    "stage4": "Stage 4\n768-d, 7x7",
    "final": "Final output\n(baseline)",
}


def extract_layer_features(frames, model, transform, out_path="data/layerwise_features.npz",
                           splits=("train", "validation"), batch_size=32, device="cpu", overwrite=False):
    """
    Run every frame of the given splits through the frozen model once and save, per stage,
    the stage output averaged over its grid. Reuses the saved file if it already exists.
    """
    assert "test" not in splits, "the test split stays held out"
    if os.path.exists(out_path) and not overwrite:
        data = np.load(out_path)
        meta = pd.DataFrame({k: data[k] for k in ["member", "identity", "sequence", "split"]})
        return {s: data[s] for s in STAGES}, meta

    keep = frames[frames["split"].isin(splits)].reset_index()   # "index" = row in the saved embeddings
    parts = {s: [] for s in STAGES}
    for start in tqdm(range(0, len(keep), batch_size), desc="Layer features"):
        paths = keep["member"].iloc[start:start + batch_size]
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.inference_mode():
            final, stages = model.forward_intermediates(batch)   # stages: NCHW; final: NHWC, normalized
        for name, out in zip(STAGES[:4], stages):
            parts[name].append(out.mean(dim=(2, 3)).cpu())
        parts["final"].append(final.mean(dim=(1, 2)).cpu())
    features = {s: torch.cat(parts[s]).numpy() for s in STAGES}

    meta = keep[["member", "identity", "sequence", "split"]]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, embedding_row=keep["index"].to_numpy(), **features,
                        **{c: meta[c].to_numpy(dtype=str) for c in meta.columns})
    return features, meta


def video_embeddings(features, meta):
    """Same as the baseline: average each video's frames, then scale to length 1."""
    groups = meta.groupby("sequence").indices
    sequences = sorted(groups)
    embs = np.stack([features[groups[s]].mean(axis=0) for s in sequences])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    videos = (meta.drop_duplicates("sequence").set_index("sequence")
              .loc[sequences, ["identity", "split"]].reset_index())
    return embs, videos


def rank1_results(video_embs, videos, query_split="validation"):
    """
    One row per query video: prediction, correct or not, the margin
    (similarity to the true bear's best training video minus the best other bear's),
    and which other bear came closest (`closest_wrong`).
    """
    train = videos.index[videos["split"] == "train"]
    queries = videos.index[videos["split"] == query_split]
    sims = cosine_similarity(video_embs[queries], video_embs[train])
    train_ids = videos.loc[train, "identity"].values
    rows = []
    for i, q in enumerate(queries):
        identity = videos.loc[q, "identity"]
        same = train_ids == identity
        rows.append({
            "sequence": videos.loc[q, "sequence"],
            "identity": identity,
            "predicted": train_ids[sims[i].argmax()],
            "margin": sims[i, same].max() - sims[i, ~same].max(),
            "closest_wrong": train_ids[~same][sims[i, ~same].argmax()],
        })
    out = pd.DataFrame(rows)
    out["correct"] = out["identity"] == out["predicted"]
    return out


def layerwise_accuracy(features, meta, combos=None, query_split="validation"):
    """
    Validation Rank-1 accuracy for every stage (and for optional combinations of stages, e.g.
    {"stage3 + final": ["stage3", "final"]}, where the similarity is the average of the stages'
    similarities). Returns the summary table and the per-video results of each stage.
    Combine centered features (`center_features`): uncentered stage-3 similarities are all about
    0.999, so in an uncentered combination the final output's similarities drown them out.
    """
    combos = combos or {}
    video = {s: video_embeddings(features[s], meta) for s in STAGES}
    per_stage = {s: rank1_results(*video[s], query_split) for s in STAGES}
    for name, parts in combos.items():
        embs = np.concatenate([video[p][0] for p in parts], axis=1) / np.sqrt(len(parts))
        per_stage[name] = rank1_results(embs, video[parts[0]][1], query_split)
    table = pd.DataFrame([{
        "stage": name,
        "accuracy": res["correct"].mean() * 100,
        "correct": int(res["correct"].sum()),
        "videos": len(res),
        "median_margin": res["margin"].median(),
    } for name, res in per_stage.items()])
    return table, per_stage


def center_features(features, meta):
    """
    Subtract the training frames' average from every stage. Raw stage outputs share a large
    common part (stage 3: every video pair has cosine about 0.999), so without this, cosine
    mostly compares small leftovers on top of that common part.
    """
    train = (meta["split"] == "train").to_numpy()
    return {s: f - f[train].mean(axis=0) for s, f in features.items()}


def _euclidean_rank1(features, meta, query_split="validation"):
    """Rank-1 accuracy matching videos by Euclidean distance between their average frames (not scaled to length 1)."""
    groups = meta.groupby("sequence").indices
    sequences = sorted(groups)
    embs = np.stack([features[groups[s]].mean(axis=0) for s in sequences])
    videos = meta.drop_duplicates("sequence").set_index("sequence").loc[sequences, ["identity", "split"]]
    train = np.where(videos["split"] == "train")[0]
    queries = np.where(videos["split"] == query_split)[0]
    dist = ((embs[queries, None, :] - embs[None, train, :]) ** 2).sum(axis=-1)
    predicted = videos["identity"].to_numpy()[train][dist.argmin(axis=1)]
    return (predicted == videos["identity"].to_numpy()[queries]).mean() * 100


def similarity_check(features, meta, query_split="validation"):
    """
    Accuracy per stage under three reasonable ways of comparing videos. If a stage only wins
    under one of them, its win depends on the comparison, not on the stage.
      cosine (as run):      the baseline's measure on the raw stage outputs
      cosine, centered:     same, after subtracting the training frames' average
      Euclidean, centered:  distance instead of angle, after subtracting the average
    """
    centered = center_features(features, meta)
    rows = []
    for s in STAGES:
        rows.append({
            "stage": s,
            "cosine (as run)": rank1_results(*video_embeddings(features[s], meta), query_split)["correct"].mean() * 100,
            "cosine, centered": rank1_results(*video_embeddings(centered[s], meta), query_split)["correct"].mean() * 100,
            "Euclidean, centered": _euclidean_rank1(centered[s], meta, query_split),
        })
    return pd.DataFrame(rows).set_index("stage")


def compare_to_baseline(per_stage, stage, baseline="final"):
    """Validation videos that `stage` gets right and the baseline gets wrong ("fixed"), and the reverse ("broken")."""
    a, b = per_stage[stage].set_index("sequence"), per_stage[baseline].set_index("sequence")
    both = a[["identity", "predicted", "correct"]].merge(
        b[["predicted", "correct"]], left_index=True, right_index=True, suffixes=(f"_{stage}", f"_{baseline}"))
    fixed = both[both[f"correct_{stage}"] & ~both[f"correct_{baseline}"]]
    broken = both[~both[f"correct_{stage}"] & both[f"correct_{baseline}"]]
    return fixed, broken


def plot_layerwise_accuracy(table, out_path="results/layerwise_accuracy.png"):
    """Bar chart of validation Rank-1 accuracy per stage; the baseline bar is highlighted."""
    white, ink, muted, grid = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
    base_color, other_color = "#2a78d6", "#c3c2b7"
    labels = [STAGE_LABELS.get(s, s.replace(" + ", "\n+ ")) for s in table["stage"]]
    colors = [base_color if s == "final" else other_color for s in table["stage"]]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=white)
    ax.set_facecolor(white)
    ax.bar(range(len(table)), table["accuracy"], color=colors, width=0.6, zorder=2)
    for x, (acc, n, total) in enumerate(zip(table["accuracy"], table["correct"], table["videos"])):
        ax.text(x, acc + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=ink)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(labels, fontsize=10, color=ink)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=10, colors=muted, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.set_ylabel("Validation Rank-1 accuracy (%)", fontsize=11, color=ink)
    ax.set_title("Which layer of MegaDescriptor identifies polar bears best?", fontsize=14,
                 fontweight="bold", color=ink, loc="left", pad=12)
    fig.text(0.01, 0.01, "Same baseline at every stage: average the 20 frames, match each validation video "
             "to its most similar training video.\nCosine on the raw stage outputs. Stage 3's lead "
             "disappears after subtracting the average frame (see similarity_check).",
             fontsize=8.5, color=muted)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=white)
    return fig, out_path


# ---------------------------------------------------------------- what the similarity space looks like

PAIR_KINDS = ["same bear, same camera", "same bear, other camera",
              "zoo-mate (other bear, same zoo)", "bear from another zoo"]


def pair_similarities(features, meta, zoos, query_split="validation"):
    """
    Cosine similarity of every validation video to every training video (the comparisons the
    baseline makes), split into the four kinds of pairs in PAIR_KINDS. Cameras are only compared
    within one bear: the same camera number is a different place for different bears.
    """
    embs, videos = video_embeddings(features, meta)
    ids = videos["identity"].to_numpy()
    cams = videos["sequence"].str.extract(r"C(\d+)T")[0].to_numpy()
    zoo = videos["identity"].map(zoos).to_numpy()
    q = np.where(videos["split"] == query_split)[0]
    t = np.where(videos["split"] == "train")[0]
    sims = embs[q] @ embs[t].T
    bear = ids[q][:, None] == ids[t][None, :]
    cam = cams[q][:, None] == cams[t][None, :]
    same_zoo = zoo[q][:, None] == zoo[t][None, :]
    masks = [bear & cam, bear & ~cam, ~bear & same_zoo, ~same_zoo]
    return {kind: sims[m] for kind, m in zip(PAIR_KINDS, masks)}


def plot_pair_similarities(groups, out_path="results/layerwise_pair_similarity.png"):
    """One histogram per kind of pair, sharing the x-axis, with the median marked."""
    white, ink, muted = "#ffffff", "#0b0b0b", "#52514e"
    colors = ["#2a78d6", "#2a78d6", "#8a8984", "#8a8984"]       # same bear: blue, other bear: gray
    lo = min(v.min() for v in groups.values())
    hi = max(v.max() for v in groups.values())
    bins = np.linspace(lo, hi, 70)

    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 6.5), sharex=True, facecolor=white)
    for ax, (kind, values), color in zip(axes, groups.items(), colors):
        ax.set_facecolor(white)
        ax.hist(values, bins=bins, density=True, color=color, zorder=2)
        med = np.median(values)
        ax.axvline(med, color=ink, linewidth=1.5, zorder=3)
        ax.text(med + 0.01, 0.95, f"median {med:.2f}", transform=ax.get_xaxis_transform(), ha="left",
                va="top", fontsize=10, fontweight="bold", color=ink, zorder=4)
        ax.text(-0.01, 0.5, f"{kind}\n{len(values):,} pairs", transform=ax.transAxes, ha="right",
                va="center", fontsize=10.5, color=ink)
        ax.set_yticks([])
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(muted)
        ax.tick_params(axis="x", labelsize=10, colors=muted)
    axes[-1].set_xlabel("Cosine similarity of a validation video to a training video", fontsize=10.5, color=ink)
    fig.text(0.01, 0.97, "Same bear, other camera: barely more similar than a zoo-mate", fontsize=14,
             fontweight="bold", color=ink, va="top")
    fig.text(0.01, 0.915, "The baseline's embedding (final output). Each row compares every validation video "
             "with every training video of that kind.", fontsize=9.5, color=muted, va="top")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.1, hspace=0.35)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=white)
    return fig, out_path


def plot_zoo_map(features, meta, zoos, names, out_path="results/layerwise_zoo_map.png", seed=0):
    """
    Every video (training and validation) placed in 2D with t-SNE (cosine), one panel per zoo:
    that zoo's bears in color, every other video in light gray. t-SNE keeps similar videos close,
    but the distances between far-apart groups mean little.
    """
    white, ink, muted, faint = "#ffffff", "#0b0b0b", "#52514e", "#dcdbd5"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    embs, videos = video_embeddings(features, meta)
    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=seed).fit_transform(embs)
    zoo = videos["identity"].map(zoos).to_numpy()
    zoo_order = list(dict.fromkeys(zoos.sort_index()))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), facecolor=white)
    for ax, z in zip(axes.ravel(), zoo_order):
        ax.set_facecolor(white)
        ax.scatter(xy[:, 0], xy[:, 1], s=5, color=faint, linewidths=0, zorder=1)
        for bear, color in zip(sorted(videos.loc[zoo == z, "identity"].unique()), colors):
            m = (videos["identity"] == bear).to_numpy()
            ax.scatter(xy[m, 0], xy[m, 1], s=14, color=color, edgecolors=white, linewidths=0.4, zorder=2,
                       label=f"{bear} {names[bear]}")
        ax.set_title(z, fontsize=11.5, fontweight="bold", color=ink, loc="left")
        ax.legend(fontsize=9.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0), ncol=3,
                  handletextpad=0.2, columnspacing=1.2, markerscale=1.6, labelcolor=ink)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(faint)
    fig.text(0.01, 0.985, "Zoo-mates often share one region; in Berlin the two bears are fully mixed", fontsize=14,
             fontweight="bold", color=ink, va="top")
    fig.text(0.01, 0.945, "Every training and validation video (final output), placed with t-SNE: similar "
             "videos end up close; distances between far-apart groups mean little.", fontsize=9.5,
             color=muted, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=white)
    return fig, out_path
