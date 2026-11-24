"""
Two bears in one crop: does a second bear in the frame cause the Tonja (002) / Hertha (003) mix-ups?

Each video (tracklet) has one label: the bear the crop follows. A second bear that walks into
the crop is not labeled, so averaging the frames mixes both bears into one video embedding.

Experiment (nothing is trained; validation videos are the queries, training videos the gallery;
the test split is never used):
1. Count the bears in every frame with Mask R-CNN (COCO animals with score >= 0.5; a bear
   found twice under two labels, e.g. "bear" and "sheep", is counted once).
2. Rebuild each video's embedding from its single-bear frames only (from the saved frame
   embeddings, so no new model passes).
3. Control: drop the same number of RANDOM frames from each video instead.
4. Compare accuracy, Tonja's and Hertha's accuracy, the mix-ups between them, and how
   separated the two bears are.
If the mix-ups drop in step 2 but not in step 3, the second bear was causing them.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor

from experiments import layerwise

BEARS = ("002", "003")      # Tonja and Hertha, Tierpark Berlin


# ---------------------------------------------------------------- 1. count the bears

def count_animals(frames, segmenter, animal_ids, out_path="data/animal_counts.csv",
                  min_score=0.5, batch_size=8, overwrite=False):
    """
    For every frame: how many animals Mask R-CNN finds with score >= min_score, and how big the
    second-largest one is (`second_box_frac`, box area / frame area; 0 if there is only one).
    Saved to a CSV and reused next time.
    """
    if os.path.exists(out_path) and not overwrite:
        counts = pd.read_csv(out_path)
        assert list(counts["member"]) == list(frames["member"]), "saved counts are for other frames"
        return counts

    rows = []
    for start in tqdm(range(0, len(frames), batch_size), desc="Counting bears"):
        paths = list(frames["member"].iloc[start:start + batch_size])
        images = [to_tensor(Image.open(p).convert("RGB")) for p in paths]
        with torch.inference_mode():
            outputs = segmenter(images)
        for path, image, out in zip(paths, images, outputs):
            is_animal = torch.tensor([int(l) in animal_ids for l in out["labels"]], dtype=torch.bool)
            ok = is_animal & (out["scores"] >= min_score)
            boxes, scores = out["boxes"][ok], out["scores"][ok]
            boxes = boxes[nms(boxes, scores, iou_threshold=0.5)]      # same bear, two labels -> once
            areas = sorted(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).tolist(), reverse=True)
            frame_area = image.shape[1] * image.shape[2]
            rows.append({"member": path, "animals": len(areas),
                         "second_box_frac": areas[1] / frame_area if len(areas) > 1 else 0.0})
    counts = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    counts.to_csv(out_path, index=False)
    return counts


def how_common(frames, counts):
    """Share of frames with 2+ animals, and of videos with at least one such frame, per bear and split."""
    f = frames.assign(two=(counts["animals"] >= 2).to_numpy())
    per_video = f.groupby(["identity", "split", "sequence"])["two"].any()
    return pd.DataFrame({
        "frames": f.groupby(["identity", "split"]).size(),
        "two_bear_frames_%": f.groupby(["identity", "split"])["two"].mean() * 100,
        "videos": per_video.groupby(["identity", "split"]).size(),
        "videos_with_two_bears": per_video.groupby(["identity", "split"]).sum(),
    }).round(1)


# ---------------------------------------------------------------- 2-4. the experiment

def _scores(frames, embeddings, use, bears, zoos=None):
    """Run the baseline using only the frames in `use` and measure the 002/003 numbers."""
    meta = frames.loc[use, ["member", "identity", "sequence", "split"]].reset_index(drop=True)
    video_embs, videos = layerwise.video_embeddings(embeddings[frames.index[use]], meta)
    res = layerwise.rank1_results(video_embs, videos)
    a, b = bears
    train = videos["split"] == "train"
    ia, ib = np.where(train & (videos["identity"] == a))[0], np.where(train & (videos["identity"] == b))[0]
    sims = video_embs @ video_embs.T
    within = lambda i: sims[np.ix_(i, i)][~np.eye(len(i), dtype=bool)].mean()
    return {
        "accuracy_%": res["correct"].mean() * 100,
        f"{a}_accuracy_%": res.loc[res["identity"] == a, "correct"].mean() * 100,
        f"{b}_accuracy_%": res.loc[res["identity"] == b, "correct"].mean() * 100,
        f"{a}->{b}_mistakes": int(((res["identity"] == a) & (res["predicted"] == b)).sum()),
        f"{b}->{a}_mistakes": int(((res["identity"] == b) & (res["predicted"] == a)).sum()),
        "gap_within_minus_between": (within(ia) + within(ib)) / 2 - sims[np.ix_(ia, ib)].mean(),
    }, res


def two_bear_experiment(frames, embeddings, counts, bears=BEARS, min_second_frac=0.0, n_random=20, seed=0):
    """
    frames: the frames to use (e.g. background_test.pick_frames, 5 per video), index = row in the
            saved embeddings; counts: animals per frame for the frames of `bears`, same order.
    min_second_frac: only count a frame as "two bears" if the second animal's box covers at least
            this share of the frame (0.1 = ignore small slivers at the edge).
    Rows of the result:
      all frames:          the baseline on these frames
      single-bear frames:  002/003 videos use only frames with one animal
      random drop:         002/003 videos drop the same number of random frames (mean of n_random)
    A video whose frames all show two bears keeps all its frames (counted in `videos_kept_whole`).
    """
    frames = frames.copy()
    is_pair = frames["identity"].isin(bears).to_numpy()
    two = np.zeros(len(frames), dtype=bool)
    two[np.where(is_pair)[0]] = ((counts["animals"] >= 2) & (counts["second_box_frac"] >= min_second_frac)).to_numpy()

    single = ~two
    whole = frames.assign(single=single).groupby("sequence")["single"].transform("sum").to_numpy() == 0
    use_single = single | whole                       # fall back to all frames if none is single

    rows, results = [], {}
    base, results["all frames"] = _scores(frames, embeddings, np.ones(len(frames), dtype=bool), bears)
    rows.append({"version": "all frames", **base})
    filt, results["single-bear frames"] = _scores(frames, embeddings, use_single, bears)
    rows.append({"version": "single-bear frames", **filt,
                 "frames_dropped": int((~use_single).sum()),
                 "videos_kept_whole": int(frames.loc[whole, "sequence"].nunique())})

    rng = np.random.default_rng(seed)
    n_drop = (frames.assign(drop=~use_single).groupby("sequence")["drop"].transform("sum")).to_numpy()
    repeats = []
    for _ in range(n_random):
        use = np.ones(len(frames), dtype=bool)
        for seq, idx in frames.reset_index().groupby("sequence").indices.items():
            k = int(n_drop[idx[0]])
            if k:
                use[rng.choice(idx, size=k, replace=False)] = False
        repeats.append(_scores(frames, embeddings, use, bears)[0])
    rows.append({"version": f"random drop (mean of {n_random})", **pd.DataFrame(repeats).mean().to_dict(),
                 "frames_dropped": int((~use_single).sum())})
    return pd.DataFrame(rows), results


# ---------------------------------------------------------------- figure

def plot_two_bear_frames(frames, counts, n=16, seed=0, out_path="results/two_bears/two_bear_frames.png"):
    """A few frames where 2+ animals were found, to check the counts by eye."""
    f = frames.reset_index(drop=True).assign(animals=counts["animals"].to_numpy())
    picks = f[f["animals"] >= 2].sample(n=min(n, int((f["animals"] >= 2).sum())), random_state=seed)
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.3 * rows), facecolor="white", squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), picks.iterrows()):
        ax.imshow(Image.open(r["member"]))
        ax.set_title(f"{r['sequence']} F{r['frame']:03d} (bear {r['identity']}): {r['animals']} animals", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor="white")
    return fig, out_path
