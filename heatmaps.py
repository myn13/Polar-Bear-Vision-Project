"""
Similarity heatmaps: which parts of each frame made the baseline match a validation video
to a training video (like the "Show, Attend and Tell" attention maps in lecture).

Why not raw attention weights: MegaDescriptor is a Swin transformer. It has no [CLS]
token and its self-attention only looks inside small local windows, so there is no
single "where is the model looking" attention map to read off like in a plain ViT.

What we use instead is exact for our baseline:
- The last Swin stage gives a 7 x 7 grid of 768-number patch features per frame.
- The frame embedding is the average of those 49 patch features.
- The video embedding is the (normalized) average of its 20 frame embeddings.
- The prediction is the training video with the highest cosine similarity.
So the similarity splits exactly into one number per patch:
    patch score = patch feature . (unit-length embedding of the other video)
and adding up all patch scores of all frames gives back the similarity (up to a
constant). Bright patches are the regions that pushed the two videos together.

The test split is never used here: everything defaults to the validation split.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from core_objective import aggregate_video_embeddings

# Chart colors (validated categorical slots 1 and 2, light mode)
SURFACE, INK, MUTED, GRID_LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CORRECT_COLOR, WRONG_COLOR = "#2a78d6", "#eb6834"


def load_frames(npz_path="data/polarbear_embeddings.npz"):
    """One row per sampled frame (in the same order as the embeddings), plus the embeddings."""
    data = np.load(npz_path)
    frames = pd.DataFrame({k: data[k] for k in ["member", "identity", "sequence", "split"]})
    frames["frame"] = frames["member"].str.extract(r"F(\d+)\.jpg$", expand=False).astype(int)
    return frames, data["embeddings"]


def match_query_videos(npz_path="data/polarbear_embeddings.npz", query_split="validation"):
    """
    Rerun the baseline and keep the details for every query video (validation by default;
    the test split stays held out):
    - match:      the most similar training video (its bear is the prediction)
    - true_best:  the most similar training video of the TRUE bear
    - other_best: the most similar training video of any OTHER bear
    margin = true_best_sim - other_best_sim. Positive means correct; the bigger, the safer.
    """
    video_embs, videos = aggregate_video_embeddings(npz_path)
    train = videos.index[videos["split"] == "train"]
    queries = videos.index[videos["split"] == query_split]
    sims = cosine_similarity(video_embs[queries], video_embs[train])
    train_ids = videos.loc[train, "identity"].values
    train_seqs = videos.loc[train, "sequence"].values

    rows = []
    for i, t in enumerate(queries):
        identity = videos.loc[t, "identity"]
        best = sims[i].argmax()
        same = np.where(train_ids == identity)[0]
        other = np.where(train_ids != identity)[0]
        best_true = same[sims[i, same].argmax()]
        best_other = other[sims[i, other].argmax()]
        rows.append({
            "sequence": videos.loc[t, "sequence"],
            "identity": identity,
            "predicted": train_ids[best],
            "match_sequence": train_seqs[best],
            "match_sim": sims[i, best],
            "true_best_sequence": train_seqs[best_true],
            "true_best_sim": sims[i, best_true],
            "other_best_sequence": train_seqs[best_other],
            "other_best_identity": train_ids[best_other],
            "other_best_sim": sims[i, best_other],
        })
    results = pd.DataFrame(rows)
    results["correct"] = results["identity"] == results["predicted"]
    results["margin"] = results["true_best_sim"] - results["other_best_sim"]
    seq_to_emb = dict(zip(videos["sequence"], video_embs))
    return results, seq_to_emb


def pick_examples(results):
    """
    Correct videos: for every bear, its most confident one (largest margin) and its
    closest call (smallest margin). Failures: every wrong video.
    """
    correct = results[results["correct"]]
    confident = correct.loc[correct.groupby("identity")["margin"].idxmax()].assign(kind="confident")
    close = correct.loc[correct.groupby("identity")["margin"].idxmin()].assign(kind="close call")
    successes = (pd.concat([confident, close]).drop_duplicates("sequence")
                 .sort_values(["identity", "kind"], ascending=[True, False]))
    failures = results[~results["correct"]].assign(kind="wrong").sort_values("margin")
    return successes, failures


# ---------------------------------------------------------------- numbers (no images needed)

def frame_stats(frames, embeddings, results, seq_to_emb):
    """
    Per query video, using only the saved embeddings:
    - top_frame_share:  share of the similarity coming from the single biggest frame
    - effective_frames: how many frames really count (1 / sum of squared shares; 20 = all equal)
    - frames_right:     share of the 20 frames that, on their own, are closest to a training
                        video of the true bear. High here but wrong overall means averaging hurt.
    """
    train = frames.loc[frames["split"] == "train", ["sequence", "identity"]].drop_duplicates()
    train_embs = np.stack([seq_to_emb[s] for s in train["sequence"]])
    train_ids = train["identity"].values

    rows = []
    for _, r in results.iterrows():
        idx = frames.index[frames["sequence"] == r["sequence"]]
        shares = frame_shares(frames, embeddings, r["sequence"], seq_to_emb[r["match_sequence"]]).values
        frame_pred = train_ids[cosine_similarity(embeddings[idx], train_embs).argmax(axis=1)]
        rows.append({
            "sequence": r["sequence"],
            "top_frame_share": shares.max(),
            "effective_frames": 1 / np.sum(shares ** 2),
            "frames_right": np.mean(frame_pred == r["identity"]),
        })
    return results.merge(pd.DataFrame(rows), on="sequence")


def confusion_pairs(results):
    """Which true bear gets mistaken for which other bear, and how often."""
    wrong = results[~results["correct"]]
    return (wrong.groupby(["identity", "predicted"]).size().rename("count")
            .reset_index().sort_values("count", ascending=False).reset_index(drop=True))


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID_LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_LINE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def plot_correct_vs_wrong(stats, out_path="results/heatmaps/analysis_correct_vs_wrong.png", seed=0):
    """
    Three panels, one dot per validation video, correct (blue) vs wrong (orange):
    margin, frames that are right on their own, and how many frames really count.
    """
    panels = [
        ("margin", "Margin\n(true bear sim - best other bear sim)", "{:+.2f}"),
        ("frames_right", "Frames right on their own\n(of 20)", "{:.0%}"),
        ("effective_frames", "Frames that really count\n(20 = all equal)", "{:.1f}"),
    ]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE)
    groups = [("Correct", stats[stats["correct"]], CORRECT_COLOR, 1),
              ("Wrong", stats[~stats["correct"]], WRONG_COLOR, 0)]
    for ax, (col, title, fmt) in zip(axes, panels):
        _style(ax)
        for label, df, color, y in groups:
            jitter = rng.uniform(-0.18, 0.18, len(df))
            ax.scatter(df[col], y + jitter, s=26, color=color, alpha=0.7,
                       edgecolor=SURFACE, linewidth=0.8, label=label)
            med = df[col].median()
            ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, linewidth=2)
            ax.text(med, y + 0.36, "median " + fmt.format(med), ha="center", fontsize=8.5, color=INK)
        if col == "margin":
            ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"Wrong ({len(groups[1][1])})", f"Correct ({len(groups[0][1])})"])
        ax.set_ylim(-0.6, 1.7)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        if col == "frames_right":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Correct vs wrong validation videos", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    return fig, out_path


# ---------------------------------------------------------------- heatmaps (need the frames)

def patch_features(model, transform, paths, device="cpu", batch_size=32):
    """Run frames through MegaDescriptor and keep the 7 x 7 grid of patch features: (N, 7, 7, 768)."""
    parts = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]])
        with torch.inference_mode():
            parts.append(model.forward_features(batch.to(device)).cpu())
    return torch.cat(parts).numpy()


def similarity_maps(patches, other_video_emb):
    """Patch score = patch feature . unit-length embedding of the other video. Shape (N, 7, 7)."""
    target = other_video_emb / np.linalg.norm(other_video_emb)
    return patches @ target


def frame_shares(frames, embeddings, sequence, other_video_emb):
    """
    How much each of the video's 20 frames adds to the similarity, as a share of the total.
    A frame's contribution is its embedding . the other video's unit embedding
    (the average of that frame's 49 patch scores).
    """
    rows = frames[frames["sequence"] == sequence].sort_values("frame")
    target = other_video_emb / np.linalg.norm(other_video_emb)
    contrib = embeddings[rows.index] @ target
    return pd.Series(contrib / contrib.sum(), index=rows.index)


def _overlay(ax, path, heat, vmax):
    """Draw a frame with its 7 x 7 heatmap upsampled to the image size on top."""
    img = np.asarray(Image.open(path).convert("RGB"))
    up = F.interpolate(torch.tensor(heat)[None, None], size=img.shape[:2],
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    ax.imshow(img)
    ax.imshow(np.clip(up, 0, None), cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_example(example, frames, embeddings, seq_to_emb, model, transform,
                 n_cols=5, device="cpu", out_dir="results/heatmaps"):
    """
    One figure per validation video, in three sections:
      1. the validation video, colored by what matched the winner
      2. the winner (most similar training video; its bear is the prediction), colored by what
         matched the validation video
      3. the runner-up from the other side, colored by what matched the validation video:
         correct -> the closest video of a WRONG bear; wrong -> the closest video of the TRUE bear
    Each section shows:
      - top left: the average of the video's frames with the average of their heatmaps on top
        (the video-level heatmap; frames are crops around the bear, so positions line up roughly)
      - top right: how much each frame adds to the similarity (dashed line = equal share)
      - below: ALL frames (4 rows x 5), each as the plain photo with its heatmap right under it
    Every heatmap in the figure uses one color scale, so brightness can be compared everywhere.
    """
    query = example["sequence"]
    if example["correct"]:
        third = (example["other_best_sequence"], query,
                 f"3. Runner-up: closest WRONG bear, train video {example['other_best_sequence']} "
                 f"(bear {example['other_best_identity']}, sim {example['other_best_sim']:.3f})")
    else:
        third = (example["true_best_sequence"], query,
                 f"3. Runner-up: closest video of the TRUE bear, train video {example['true_best_sequence']} "
                 f"(bear {example['identity']}, sim {example['true_best_sim']:.3f})")
    sections = [(query, example["match_sequence"],
                 f"1. Validation video {query} (bear {example['identity']})"),
                (example["match_sequence"], query,
                 f"2. Winner: train video {example['match_sequence']} "
                 f"(bear {example['predicted']}, sim {example['match_sim']:.3f})"),
                third]

    # Compute every map first so the whole figure shares one color scale
    panels = []
    for seq, other, label in sections:
        video = frames[frames["sequence"] == seq].sort_values("frame")
        maps = similarity_maps(patch_features(model, transform, list(video["member"]), device), seq_to_emb[other])
        shares = frame_shares(frames, embeddings, seq, seq_to_emb[other]).values
        panels.append((video, maps, shares, label))
    vmax = max(np.percentile(m, 99) for _, m, _, _ in panels)

    n_rows = int(np.ceil(max(len(v) for v, _, _, _ in panels) / n_cols))
    col_w = 2.5                       # inches per frame column
    frame_h = col_w / 2               # frames are 2:1
    section_h = 2.75 + 0.7 + n_rows * (2 * frame_h + 0.35)
    fig = plt.figure(figsize=(n_cols * col_w, len(panels) * (section_h + 0.5) + 0.6), facecolor="white")
    outer = fig.add_gridspec(len(panels), 1, hspace=0.12)

    for i, (video, maps, shares, label) in enumerate(panels):
        # row 0: summary, row 1: spacer for the bar labels, then plain/heatmap row pairs
        inner = outer[i].subgridspec(2 + 2 * n_rows, n_cols, hspace=0.3, wspace=0.06,
                                     height_ratios=[2.2, 0.45] + [1, 1] * n_rows)
        images = [np.asarray(Image.open(p).convert("RGB")) for p in video["member"]]

        # Video-level view: average frame + average heatmap
        ax_avg = fig.add_subplot(inner[0, :2])
        avg_img = np.mean(images, axis=0).astype(np.uint8)
        up = F.interpolate(torch.tensor(maps.mean(axis=0))[None, None], size=avg_img.shape[:2],
                           mode="bilinear", align_corners=False)[0, 0].numpy()
        ax_avg.imshow(avg_img)
        ax_avg.imshow(np.clip(up, 0, None), cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
        ax_avg.set_xticks([])
        ax_avg.set_yticks([])
        ax_avg.set_title(f"{label}\nAverage of all {len(images)} frames + average heatmap",
                         loc="left", fontsize=10.5, fontweight="bold", color=INK)

        # How the frames add up
        ax_bar = fig.add_subplot(inner[0, 2:])
        _style(ax_bar)
        ax_bar.grid(False, axis="x")
        ax_bar.grid(True, axis="y", color=GRID_LINE, linewidth=0.8)
        ax_bar.bar(range(len(shares)), shares * 100, color=CORRECT_COLOR, width=0.75)
        ax_bar.axhline(100 / len(shares), color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax_bar.set_xticks(range(len(shares)))
        ax_bar.set_xticklabels([f"F{f:03d}" for f in video["frame"]], rotation=90, fontsize=7)
        ax_bar.set_title(f"How the {len(shares)} frames add up: % of the similarity per frame (dashed = equal share)",
                         loc="left", fontsize=9.5, color=INK)

        # All frames: plain photo, heatmap right under it
        for k, (img, (_, row)) in enumerate(zip(images, video.iterrows())):
            r, c = divmod(k, n_cols)
            ax_plain = fig.add_subplot(inner[2 + 2 * r, c])
            ax_plain.imshow(img)
            ax_plain.set_title(f"F{row['frame']:03d}   {shares[k]:.0%}", fontsize=8.5, color=INK, pad=3)
            ax_heat = fig.add_subplot(inner[3 + 2 * r, c])
            _overlay(ax_heat, row["member"], maps[k], vmax)
            for ax in (ax_plain, ax_heat):
                ax.set_xticks([])
                ax.set_yticks([])

    status = {"confident": "CORRECT (confident)", "close call": "CORRECT (close call)"}.get(example["kind"], "WRONG")
    fig.suptitle(f"{status}: validation video {query} is bear {example['identity']}, "
                 f"predicted bear {example['predicted']}  |  margin {example['margin']:+.3f}",
                 x=0.01, ha="left", y=1 - 0.2 / fig.get_figheight(), fontsize=14, fontweight="bold",
                 color="#1a7f37" if example["correct"] else "#c62828")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.1 / fig.get_figheight(), bottom=0.01)

    os.makedirs(out_dir, exist_ok=True)
    prefix = {"confident": "success_confident", "close call": "success_close"}.get(example["kind"], "failure")
    path = os.path.join(out_dir, f"{prefix}_{query}.jpg")
    fig.savefig(path, dpi=100, facecolor="white", pil_kwargs={"quality": 85})
    return fig, path


# ---------------------------------------------------------------- top-patch gallery (plain images)

GRID = 7  # the last Swin stage is a 7 x 7 grid of patches


def top_patches(results, frames, seq_to_emb, model, transform, k=3, device="cpu"):
    """
    For every query video, score all 20 x 49 patches toward the video it was matched to
    (the winner) and keep the k best, each from a different frame.
    For wrong videos, also keep the k best patches toward the closest video of the TRUE bear.
    Returns one row per kept patch: where it is (frame, grid row/col) and its score.
    """
    rows = []
    for _, r in results.iterrows():
        video = frames[frames["sequence"] == r["sequence"]].sort_values("frame")
        patches = patch_features(model, transform, list(video["member"]), device)
        targets = [("winner", r["match_sequence"])]
        if not r["correct"]:
            targets.append(("true_bear", r["true_best_sequence"]))
        for target, other in targets:
            maps = similarity_maps(patches, seq_to_emb[other])            # (20, 7, 7)
            best_cell = maps.reshape(len(maps), -1).argmax(axis=1)        # best cell per frame
            best_score = maps.reshape(len(maps), -1).max(axis=1)
            for rank, f in enumerate(np.argsort(-best_score)[:k], start=1):
                row, col = divmod(int(best_cell[f]), GRID)
                rows.append({
                    "sequence": r["sequence"], "identity": r["identity"], "predicted": r["predicted"],
                    "correct": r["correct"], "target": target, "rank": rank,
                    "member": video["member"].iloc[f], "frame": video["frame"].iloc[f],
                    "grid_row": row, "grid_col": col, "score": best_score[f],
                })
    return pd.DataFrame(rows)


def _cell_box(img_w, img_h, row, col, context=1.0):
    """Pixel box of one grid cell in the original image, grown by `context` of a cell on each side."""
    cw, ch = img_w / GRID, img_h / GRID
    x0, y0 = (col - context) * cw, (row - context) * ch
    x1, y1 = (col + 1 + context) * cw, (row + 1 + context) * ch
    return max(0, int(x0)), max(0, int(y0)), min(img_w, int(round(x1))), min(img_h, int(round(y1)))


def plot_patch_gallery(tiles, title, out_path, n_cols=8):
    """
    Plain-image gallery: for each row of `tiles`, the top patch cut from the original frame
    (enlarged, with one cell of context on each side), and below it the whole frame with a
    box showing where the patch came from. No heatmap colors, so blur and darkness stay visible.
    """
    from matplotlib.patches import Rectangle

    n = len(tiles)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(2.0 * n_cols, 2.55 * n_rows + 0.7), facecolor=SURFACE)
    grid = fig.add_gridspec(2 * n_rows, n_cols, height_ratios=[1.6, 0.8] * n_rows,
                            hspace=0.08, wspace=0.08)
    for i, (_, t) in enumerate(tiles.iterrows()):
        r, c = divmod(i, n_cols)
        img = Image.open(t["member"]).convert("RGB")
        box = _cell_box(img.width, img.height, t["grid_row"], t["grid_col"])
        ax_tile = fig.add_subplot(grid[2 * r, c])
        ax_tile.imshow(img.crop(box))
        ax_full = fig.add_subplot(grid[2 * r + 1, c])
        ax_full.imshow(img)
        cw, ch = img.width / GRID, img.height / GRID
        ax_full.add_patch(Rectangle((t["grid_col"] * cw, t["grid_row"] * ch), cw, ch,
                                    fill=False, edgecolor="#eda100", linewidth=1.6))
        for ax in (ax_tile, ax_full):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID_LINE)
        who = f"bear {t['identity']}" if t["correct"] else f"bear {t['identity']} -> {t['predicted']}"
        ax_tile.set_title(f"{who}\n{t['sequence']} F{t['frame']:03d}", fontsize=7.5, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.7 / fig.get_figheight())
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=90, facecolor=SURFACE, bbox_inches="tight")
    return fig, out_path


# ---------------------------------------------------------------- checks that the heatmap math is exact

def verify_heatmap_math(results, frames, embeddings, seq_to_emb, model, transform, sequences, device="cpu"):
    """
    For each query video in `sequences`, and for its winner and runner-up (the other two
    rows of a figure), recompute everything from the images and compare:
    - embedding_vs_grid_mean: model output vs the average of the 7 x 7 patch grid (should be ~0)
    - saved_vs_recomputed:    saved frame embeddings vs a fresh run (should be ~1e-5 or less)
    - cosine_from_patches:    sum of all 20 x 49 patch scores / (980 * |average frame embedding|)
    - cosine_baseline:        the cosine similarity the baseline used to decide
    - share_diff:             frame shares from the saved embeddings vs from the patch scores
    """
    rows = []
    for seq in sequences:
        r = results[results["sequence"] == seq].iloc[0]
        third = r["other_best_sequence"] if r["correct"] else r["true_best_sequence"]
        third_sim = r["other_best_sim"] if r["correct"] else r["true_best_sim"]
        for role, video, other, cosine in [("validation video", seq, r["match_sequence"], r["match_sim"]),
                                           ("winner", r["match_sequence"], seq, r["match_sim"]),
                                           ("runner-up", third, seq, third_sim)]:
            vid = frames[frames["sequence"] == video].sort_values("frame")
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in vid["member"]])
            with torch.inference_mode():
                grid = model.forward_features(batch.to(device)).cpu().numpy()   # (20, 7, 7, 768)
                emb = model(batch.to(device)).cpu().numpy()                     # (20, 768)
            maps = similarity_maps(grid, seq_to_emb[other])                     # (20, 7, 7)
            n_cells = maps.size
            avg_frame_norm = np.linalg.norm(embeddings[vid.index].mean(axis=0))
            from_patches = maps.reshape(len(maps), -1).sum(axis=1)
            rows.append({
                "query": seq, "row": role, "video": video,
                "embedding_vs_grid_mean": np.abs(emb - grid.mean(axis=(1, 2))).max(),
                "saved_vs_recomputed": np.abs(emb - embeddings[vid.index]).max(),
                "cosine_from_patches": maps.sum() / (n_cells * avg_frame_norm),
                "cosine_baseline": cosine,
                "share_diff": np.abs(from_patches / from_patches.sum()
                                     - frame_shares(frames, embeddings, video, seq_to_emb[other]).values).max(),
                "negative_patches": (maps < 0).mean(),
            })
    out = pd.DataFrame(rows)
    out["cosine_diff"] = (out["cosine_from_patches"] - out["cosine_baseline"]).abs()
    return out


def check_grid_orientation(model, transform, paths, seed=0):
    """
    Checks that heatmaps are drawn in the right place (no flip, no transpose, correct stretch
    from the 256 x 128 frame to the 7 x 7 grid). For every one of the 49 cells, cover that
    cell's area of the ORIGINAL frame with random noise and measure how much each of the 49
    patch features changes.
    Returns, per layout, the average change (z-score) at the cell that layout says was covered.
    Only the layout we draw ("as drawn") should stand out. Neighbors also change a bit because
    Swin's shifted windows mix each cell with its surroundings, so a hot cell means
    "this region and its immediate surroundings".
    """
    layouts = {"as drawn": lambda r, c: (r, c),
               "left-right flip": lambda r, c: (r, GRID - 1 - c),
               "up-down flip": lambda r, c: (GRID - 1 - r, c),
               "transposed": lambda r, c: (c, r)}
    rng = np.random.default_rng(seed)
    z_at = {k: [] for k in layouts}
    ranks = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        pixels = np.array(img)
        with torch.inference_mode():
            base = model.forward_features(transform(img)[None])[0].numpy()
        for row in range(GRID):
            for col in range(GRID):
                covered = pixels.copy()
                x0, y0, x1, y1 = _cell_box(img.width, img.height, row, col, context=0)
                covered[y0:y1, x0:x1] = rng.integers(0, 256, (y1 - y0, x1 - x0, 3))
                with torch.inference_mode():
                    feats = model.forward_features(transform(Image.fromarray(covered))[None])[0].numpy()
                change = np.linalg.norm(feats - base, axis=-1)
                z = (change - change.mean()) / change.std()
                ranks.append(int((change > change[row, col]).sum()) + 1)
                for k, where in layouts.items():
                    z_at[k].append(z[where(row, col)])
    ranks = np.array(ranks)
    print(f"{len(ranks)} covered cells: the covered cell changed the most in {np.mean(ranks == 1):.0%}, "
          f"and was in the top 3 in {np.mean(ranks <= 3):.0%}")
    return pd.DataFrame({"layout": list(layouts), "avg_change_z": [np.mean(z_at[k]) for k in layouts]})
