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


def load_frames(npz_path="data/polarbear_embeddings.npz"):
    """One row per sampled frame (in the same order as the embeddings), plus the embeddings."""
    data = np.load(npz_path)
    frames = pd.DataFrame({k: data[k] for k in ["member", "identity", "sequence", "split"]})
    frames["frame"] = frames["member"].str.extract(r"F(\d+)\.jpg$", expand=False).astype(int)
    return frames, data["embeddings"]


def match_query_videos(npz_path="data/polarbear_embeddings.npz", query_split="validation"):
    """
    Rerun the baseline and keep the details for every query video (validation by default;
    the test split stays held out): the predicted bear, the matched training video, and the
    closest training video of the true bear.
    """
    video_embs, videos = aggregate_video_embeddings(npz_path)
    train = videos.index[videos["split"] == "train"]
    queries = videos.index[videos["split"] == query_split]
    sims = cosine_similarity(video_embs[queries], video_embs[train])
    train_ids = videos.loc[train, "identity"].values
    train_seqs = videos.loc[train, "sequence"].values

    rows = []
    for i, t in enumerate(queries):
        best = sims[i].argmax()
        same_bear = np.where(train_ids == videos.loc[t, "identity"])[0]
        best_true = same_bear[sims[i, same_bear].argmax()]
        rows.append({
            "sequence": videos.loc[t, "sequence"],
            "identity": videos.loc[t, "identity"],
            "predicted": train_ids[best],
            "match_sequence": train_seqs[best],
            "match_sim": sims[i, best],
            "true_best_sequence": train_seqs[best_true],
            "true_best_sim": sims[i, best_true],
        })
    results = pd.DataFrame(rows)
    results["correct"] = results["identity"] == results["predicted"]
    seq_to_emb = dict(zip(videos["sequence"], video_embs))
    return results, seq_to_emb


def pick_examples(results, n_success=5, seed=0):
    """n_success correct query videos (each from a different bear) and every failed one."""
    correct = results[results["correct"]]
    one_per_bear = correct.groupby("identity").sample(1, random_state=seed)
    successes = one_per_bear.sample(n_success, random_state=seed).sort_values("identity")
    failures = results[~results["correct"]].sort_values("identity")
    return successes, failures


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


def shown_frames(frames, sequence, n_show):
    """n_show evenly spaced frames (of the 20 sampled) of one video."""
    rows = frames[frames["sequence"] == sequence].sort_values("frame")
    keep = np.linspace(0, len(rows) - 1, min(n_show, len(rows))).round().astype(int)
    return rows.iloc[keep]


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
                 n_show=8, device="cpu", out_dir="results/heatmaps"):
    """
    One figure per query (validation) video. Rows:
      1. the validation video, colored by what matched the predicted training video
      2. the predicted training video, colored by what matched the validation video
      3. (failures only) the closest training video of the true bear, colored the same way
    All rows share one color scale so brightness can be compared across rows.
    The number above each frame is that frame's share of the video's total similarity.
    """
    query = example["sequence"]
    rows = [(query, example["match_sequence"],
             f"Validation video {query}\n(bear {example['identity']})"),
            (example["match_sequence"], query,
             f"Matched train video\n{example['match_sequence']}\n(bear {example['predicted']}, sim {example['match_sim']:.3f})")]
    if not example["correct"]:
        rows.append((example["true_best_sequence"], query,
                     f"Closest video of true bear\n{example['true_best_sequence']}\n(bear {example['identity']}, sim {example['true_best_sim']:.3f})"))

    # Compute every map first so all rows can share one color scale
    panels = []
    for seq, other, label in rows:
        shown = shown_frames(frames, seq, n_show)
        maps = similarity_maps(patch_features(model, transform, list(shown["member"]), device), seq_to_emb[other])
        shares = frame_shares(frames, embeddings, seq, seq_to_emb[other]).loc[shown.index]
        panels.append((shown, maps, shares.values, label))
    vmax = max(np.percentile(m, 99) for _, m, _, _ in panels)

    fig, axes = plt.subplots(len(panels), n_show, figsize=(2.1 * n_show, 1.45 * len(panels) + 0.9),
                             squeeze=False)
    for r, (shown, maps, shares, label) in enumerate(panels):
        for c in range(n_show):
            ax = axes[r, c]
            if c >= len(shown):
                ax.axis("off")
                continue
            _overlay(ax, shown["member"].iloc[c], maps[c], vmax)
            ax.set_title(f"F{shown['frame'].iloc[c]:03d}  {shares[c]:.0%}", fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=10)

    status = "CORRECT" if example["correct"] else "WRONG"
    fig.suptitle(f"{status}: validation video {query} is bear {example['identity']}, "
                 f"predicted bear {example['predicted']}", fontsize=12, fontweight="bold",
                 color="#1a7f37" if example["correct"] else "#c62828")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{'success' if example['correct'] else 'failure'}_{query}.png")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig, path
