"""
Bear vs background: does the frozen MegaDescriptor recognize each polar bear from the bear
itself, or from the enclosure behind it?

Nothing is trained:
1. Find the bear in every frame with an off-the-shelf segmentation model (torchvision's
   Mask R-CNN trained on COCO, which has a "bear" class).
2. Make edited copies of every frame (gray is the model's "zero" input after normalization):
     bear only:        everything except the bear painted gray
     background only:  the bear (plus a small margin) painted gray
     outline only:     no colors or texture at all, just the bear's shape (a control)
3. Run the exact baseline on every version: the same edit on the validation videos (queries)
   and on the training videos (gallery), the same 5 frames per video.

How to read it (chance = 1 in 13 bears = 7.7%):
- bear only close to the original and background only near chance: the model uses the bear.
- background only well above chance: the enclosure alone gives the bear away.
- outline only checks the gray hole: the hole in "background only" still has the bear's shape,
  so if outline only also scores high, shape (not the enclosure) could explain that score.
Gray areas look unnatural to the model, so every edited version loses some accuracy from the
edit itself. Compare the versions with each other, not only with the original.

The test split is never used here.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms.functional import to_tensor

from experiments import layerwise
from experiments.zoo_analysis import load_zoos  # noqa: F401  (bt.load_zoos() in the notebook)

VERSIONS = ["original", "bear_only", "background_only", "outline_only"]
VERSION_LABELS = {
    "original": "Original\n(5 frames)",
    "bear_only": "Bear only",
    "background_only": "Background only",
    "outline_only": "Outline only\n(control)",
}
GRAY = 128                      # becomes 0 after MegaDescriptor's normalization
OUTLINE_BEAR = 230              # outline only: light bear shape on gray
ANIMALS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
CHANCE = 100 / 13


def pick_frames(frames, per_video=5, splits=("train", "validation")):
    """
    The same evenly spread frames of every training and validation video for every version
    (5 of the 20 sampled frames). The index is the row in the saved baseline embeddings.
    """
    assert "test" not in splits, "the test split stays held out"
    keep = frames[frames["split"].isin(splits)].sort_values(["sequence", "frame"])
    return keep.groupby("sequence").nth(list(range(0, 20, 20 // per_video)))


# ---------------------------------------------------------------- 1. find the bear

def load_segmenter(device="cpu", min_size=256, max_objects=10):
    """
    Mask R-CNN (COCO). Frames are 256 x 128, so they are upscaled 2x before detection
    (min_size=256). On 40 test frames this gave the same masks as 3x upscaling
    (median overlap 0.96) in 30% less time. Only the `max_objects` most confident objects
    get a mask, which is enough for one or two bears.
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=min_size, box_detections_per_img=max_objects,
                                     box_score_thresh=0.2, rpn_post_nms_top_n_test=300)
    categories = weights.meta["categories"]
    animal_ids = {categories.index(a): a for a in ANIMALS}
    return model.to(device).eval(), animal_ids


def _bear_mask(output, animal_ids, image_shape, min_score):
    """Union of all animals found with score >= min_score (always including the best one)."""
    labels = [int(l) for l in output["labels"]]
    scores = output["scores"]
    animal = [i for i, l in enumerate(labels) if l in animal_ids]
    if not animal:
        return np.zeros(image_shape, dtype=bool), "none", 0.0
    best = max(animal, key=lambda i: float(scores[i]))
    use = [i for i in animal if float(scores[i]) >= min_score] or [best]
    mask = (output["masks"][use, 0] > 0.5).any(dim=0).cpu().numpy()
    return mask, animal_ids[labels[best]], float(scores[best])


def segment_bears(keep, segmenter, animal_ids, out_path="data/background_masks.npz",
                  batch_size=8, min_score=0.5, device="cpu", overwrite=False):
    """
    One bear mask per frame (True = bear), plus a table with the best animal label, its
    score, and how much of the frame is bear. Saved and reused next time.
    """
    if os.path.exists(out_path) and not overwrite:
        data = np.load(out_path)
        assert list(data["member"]) == list(keep["member"]), "saved masks are for other frames"
        masks = np.unpackbits(data["masks"], axis=-1, count=int(data["width"])).astype(bool)
        info = pd.DataFrame({"member": data["member"], "label": data["label"], "score": data["score"]})
        info["coverage"] = masks.mean(axis=(1, 2))
        return masks, info

    masks, rows = [], []
    for start in tqdm(range(0, len(keep), batch_size), desc="Finding bears"):
        paths = list(keep["member"].iloc[start:start + batch_size])
        images = [to_tensor(Image.open(p).convert("RGB")) for p in paths]
        with torch.inference_mode():
            outputs = segmenter([im.to(device) for im in images])
        for path, image, output in zip(paths, images, outputs):
            mask, label, score = _bear_mask(output, animal_ids, image.shape[1:], min_score)
            masks.append(mask)
            rows.append({"member": path, "label": label, "score": score})
    masks = np.stack(masks)
    info = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, masks=np.packbits(masks, axis=-1), width=masks.shape[-1],
                        member=info["member"].to_numpy(dtype=str), label=info["label"].to_numpy(dtype=str),
                        score=info["score"].to_numpy())
    info["coverage"] = masks.mean(axis=(1, 2))
    return masks, info


def usable_frames(info, min_score=0.5, min_coverage=0.02):
    """Frames where a bear was found with confidence. The others are left out of EVERY version."""
    return ((info["score"] >= min_score) & (info["coverage"] >= min_coverage)).to_numpy()


# ---------------------------------------------------------------- 2. edit the frames

def _grow(mask, pixels):
    """Make the mask a few pixels bigger so no bear fur is left at its edge."""
    m = torch.from_numpy(mask).float()[None, None]
    return F.max_pool2d(m, 2 * pixels + 1, stride=1, padding=pixels)[0, 0].numpy() > 0


def edit_frame(image, mask, version, margin=4):
    """One version of a frame (PIL image in, PIL image out)."""
    if version == "original":
        return image
    pixels = np.asarray(image)
    if version == "outline_only":
        shape = np.where(mask, OUTLINE_BEAR, GRAY).astype(np.uint8)
        return Image.fromarray(np.repeat(shape[..., None], 3, axis=2))
    keep = {"bear_only": mask, "background_only": ~_grow(mask, margin)}[version]
    return Image.fromarray(np.where(keep[..., None], pixels, GRAY).astype(np.uint8))


# ---------------------------------------------------------------- 3. run the baseline on each version

def version_embeddings(keep, masks, model, transform, out_path="data/background_embeddings.npz",
                       batch_size=32, device="cpu", overwrite=False):
    """MegaDescriptor embedding of every frame in every version, exactly as the baseline computes them."""
    if os.path.exists(out_path) and not overwrite:
        data = np.load(out_path)
        assert list(data["member"]) == list(keep["member"]), "saved embeddings are for other frames"
        return {v: data[v] for v in VERSIONS}

    parts = {v: [] for v in VERSIONS}
    for start in tqdm(range(0, len(keep), batch_size), desc="Embedding versions"):
        paths = keep["member"].iloc[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in paths]
        for version in VERSIONS:
            batch = torch.stack([transform(edit_frame(im, masks[start + j], version))
                                 for j, im in enumerate(images)])
            with torch.inference_mode():
                parts[version].append(model(batch.to(device)).cpu())
    embs = {v: torch.cat(parts[v]).numpy() for v in VERSIONS}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, member=keep["member"].to_numpy(dtype=str), **embs)
    return embs


def background_accuracy(embs, keep, usable, zoos, query_split="validation"):
    """
    Validation Rank-1 accuracy per version, using only the usable frames (the same frames in
    every version). Also how often the model leans toward a bear from the SAME ZOO:
      zoo_mate_mistakes: wrong predictions that picked a zoo-mate
      closest_wrong_zoo_mate_%: videos whose most similar wrong bear is a zoo-mate
                                (chance is about 10%: 1 or 2 zoo-mates out of 12 other bears)
    Returns the summary table and the per-video results of each version.
    """
    meta = keep.loc[usable, ["member", "identity", "sequence", "split"]].reset_index(drop=True)
    per_version = {}
    for version in VERSIONS:
        video_embs, videos = layerwise.video_embeddings(embs[version][usable], meta)
        res = layerwise.rank1_results(video_embs, videos, query_split)
        own_zoo = res["identity"].map(zoos)
        res["predicted_zoo_mate"] = ~res["correct"] & (res["predicted"].map(zoos) == own_zoo)
        res["closest_wrong_zoo_mate"] = res["closest_wrong"].map(zoos) == own_zoo
        per_version[version] = res
    table = pd.DataFrame([{
        "version": v,
        "accuracy": res["correct"].mean() * 100,
        "correct": int(res["correct"].sum()),
        "videos": len(res),
        "median_margin": res["margin"].median(),
        "zoo_mate_mistakes": f"{int(res['predicted_zoo_mate'].sum())} of {int((~res['correct']).sum())}",
        "closest_wrong_zoo_mate_%": res["closest_wrong_zoo_mate"].mean() * 100,
    } for v, res in per_version.items()])
    return table, per_version


def per_bear_accuracy(per_version):
    """Validation accuracy of each version for each bear (some enclosures may be easier to recognize)."""
    return pd.DataFrame({v: res.groupby("identity")["correct"].mean() * 100
                         for v, res in per_version.items()}).round(1)


# ---------------------------------------------------------------- figures

def plot_examples(keep, masks, info, usable, n=6, seed=0, out_path="results/background/examples.png"):
    """A few frames from different bears in every version, to check the masks by eye."""
    candidates = keep.assign(row=np.arange(len(keep)), usable=usable)
    picks = (candidates[candidates["usable"]].sample(frac=1, random_state=seed)
             .groupby("identity").head(1).sort_values("identity").head(n))
    fig, axes = plt.subplots(len(picks), len(VERSIONS), figsize=(2.6 * len(VERSIONS), 1.45 * len(picks)),
                             facecolor="white")
    for r, (_, frame) in enumerate(picks.iterrows()):
        image = Image.open(frame["member"]).convert("RGB")
        for c, version in enumerate(VERSIONS):
            ax = axes[r, c]
            ax.imshow(edit_frame(image, masks[frame["row"]], version))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(VERSION_LABELS[version].split("\n")[0], fontsize=11, color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"bear {frame['identity']}", fontsize=9, color="#52514e")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    return fig, out_path


def plot_background_accuracy(table, out_path="results/background/accuracy.png"):
    """Bar chart of validation Rank-1 accuracy per version, with the chance level marked."""
    white, ink, muted, grid = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
    base_color, other_color = "#2a78d6", "#c3c2b7"
    colors = [base_color if v == "original" else other_color for v in table["version"]]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=white)
    ax.set_facecolor(white)
    ax.bar(range(len(table)), table["accuracy"], color=colors, width=0.6, zorder=2)
    for x, acc in enumerate(table["accuracy"]):
        ax.text(x, acc + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=ink)
    ax.axhline(CHANCE, color=muted, linestyle="--", linewidth=1, zorder=3)
    ax.text(len(table) - 0.55, CHANCE + 1.2, f"chance ({CHANCE:.1f}%)", ha="right", va="bottom",
            fontsize=9, color=muted)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([VERSION_LABELS[v] for v in table["version"]], fontsize=10, color=ink)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=10, colors=muted, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.set_ylabel("Validation Rank-1 accuracy (%)", fontsize=11, color=ink)
    ax.set_title("Does the model recognize the bear or the background?", fontsize=14,
                 fontweight="bold", color=ink, loc="left", pad=12)
    fig.text(0.01, 0.01, "Same edit on validation queries and training gallery; 5 frames per video; "
             "frozen MegaDescriptor-S-224.", fontsize=8.5, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=white)
    return fig, out_path


# ---------------------------------------------------------------- reviewing every mask by eye

REVIEW_DIM = 0.33                   # background brightness in the review sheets
REVIEW_EDGE = (235, 104, 52)        # orange line along the mask's edge
REVIEW_BAD = (227, 73, 72)          # red border: a frame step 3 would throw out
REVIEW_DARK = 70                    # frames darker than this (mean pixel) are brightened for viewing


def spotlight(image, mask, edge_px=2):
    """The frame with the background dimmed and the mask's edge drawn, so the mask stands out."""
    pixels = np.asarray(image.convert("RGB")).astype(np.float32)
    out = np.where(mask[..., None], pixels, pixels * REVIEW_DIM)
    edge = mask & _grow(~mask, edge_px)                     # mask pixels next to the background
    out[edge] = REVIEW_EDGE
    return Image.fromarray(out.astype(np.uint8))


def _review_fonts():
    import matplotlib
    from PIL import ImageFont
    folder = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    return (ImageFont.truetype(os.path.join(folder, "DejaVuSans.ttf"), 12),
            ImageFont.truetype(os.path.join(folder, "DejaVuSans-Bold.ttf"), 14),
            ImageFont.truetype(os.path.join(folder, "DejaVuSans-Bold.ttf"), 20))


def _frame_tile(row, mask, font, width=256, height=128, caption_h=18, with_video=False):
    """One frame of a review sheet: caption on top, spotlight image below, red border if thrown out."""
    from PIL import ImageDraw, ImageOps
    image = Image.open(row["member"]).convert("RGB")
    dark = np.asarray(image).mean() < REVIEW_DARK
    if dark:                                    # for viewing only: the test itself uses the original frame
        image = ImageOps.autocontrast(image, cutoff=1)
    tile = Image.new("RGB", (width, height + caption_h), "white")
    tile.paste(spotlight(image, mask).resize((width, height)), (0, caption_h))
    draw = ImageDraw.Draw(tile)
    color = REVIEW_BAD if not row["usable"] else (11, 11, 11)
    who = f"{row['sequence']} " if with_video else ""
    draw.text((2, 2), f"{who}F{row['frame']:03d}  {row['label']} {row['score']:.2f}  {row['coverage'] * 100:.0f}%"
              + ("  *" if dark else ""), fill=color, font=font)
    if not row["usable"]:
        draw.rectangle([0, caption_h, width - 1, caption_h + height - 1], outline=REVIEW_BAD, width=4)
    return tile


def mask_review_sheets(keep, masks, info, usable, names, out_dir="results/background/mask_review",
                       videos_per_page=20):
    """
    Every mask, for checking by eye. One set of pages per bear; each row is one video with its
    5 frames. In each frame the mask is at full brightness, the background is dimmed, and an
    orange line marks the mask's edge. Caption: frame number, the animal Mask R-CNN saw, its
    score, and how much of the frame the mask covers. A red border marks frames that step 3
    throws out. Also writes masks.csv (one row per frame, with an empty `bad_mask` column to
    fill in) and check_*.jpg pages (frames to check first: thrown out, very large, very small masks).
    """
    from PIL import ImageDraw
    font, bold, title = _review_fonts()
    os.makedirs(out_dir, exist_ok=True)
    df = keep.reset_index(drop=True)[["member", "identity", "sequence", "split", "frame"]].assign(
        row=np.arange(len(keep)), label=info["label"].to_numpy(), score=info["score"].to_numpy(),
        coverage=info["coverage"].to_numpy(), usable=usable)
    W, H, CAP, GAP, LEFT, TOP = 256, 128, 18, 8, 190, 70
    pages = []
    df["page"] = ""
    for bear, bear_df in df.groupby("identity"):
        sequences = sorted(bear_df["sequence"].unique())
        n_pages = int(np.ceil(len(sequences) / videos_per_page))
        for p in range(n_pages):
            chunk = sequences[p * videos_per_page:(p + 1) * videos_per_page]
            sheet = Image.new("RGB", (LEFT + 5 * (W + GAP), TOP + len(chunk) * (H + CAP + GAP)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((10, 8), f"Bear {bear} {names[bear]}  -  page {p + 1} of {n_pages}", fill=(11, 11, 11), font=title)
            draw.text((10, 40), "Bright = mask (bear), dimmed = background, orange line = mask edge. "
                      "Caption: frame, animal Mask R-CNN saw, score, % of frame masked. Red border = thrown out. * = dark frame brightened for viewing only.",
                      fill=(82, 81, 78), font=font)
            name = f"bear{bear}_p{p + 1:02d}.jpg"
            for r, seq in enumerate(chunk):
                y = TOP + r * (H + CAP + GAP)
                video = bear_df[bear_df["sequence"] == seq].sort_values("frame")
                draw.text((10, y + CAP + 40), seq, fill=(11, 11, 11), font=bold)
                draw.text((10, y + CAP + 62), video["split"].iloc[0], fill=(82, 81, 78), font=font)
                for c, (_, row) in enumerate(video.iterrows()):
                    sheet.paste(_frame_tile(row, masks[row["row"]], font), (LEFT + c * (W + GAP), y))
                df.loc[video.index, "page"] = name
            sheet.save(os.path.join(out_dir, name), quality=88)
            pages.append(name)

    # frames to check first, one set of pages per reason (the score is about the animal's name,
    # not the outline, so a low score alone is not a reason)
    checks = {
        "thrown_out": ("thrown out in step 3 (no animal found, or score below 0.5)", df[~df["usable"]]),
        "large_mask": ("mask covers over 70% of the frame (may spill onto the background)",
                       df[df["usable"] & (df["coverage"] > 0.7)]),
        "small_mask": ("mask covers under 15% of the frame (may miss part of the bear)",
                       df[df["usable"] & (df["coverage"] < 0.15)]),
    }
    cols, per_page = 6, 120
    check_pages = []
    for key, (what, subset) in checks.items():
        n_pages = max(1, int(np.ceil(len(subset) / per_page)))
        for p in range(n_pages):
            chunk = subset.iloc[p * per_page:(p + 1) * per_page]
            rows = max(1, int(np.ceil(len(chunk) / cols)))
            sheet = Image.new("RGB", (cols * (W + GAP) + GAP, TOP + rows * (H + CAP + GAP)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((10, 8), f"Check first: {what}", fill=(11, 11, 11), font=title)
            draw.text((10, 40), f"{len(subset)} frames, page {p + 1} of {n_pages}, sorted by bear. "
                      "Caption: video, frame, animal Mask R-CNN saw, score, % of frame masked. * = dark frame brightened for viewing only.",
                      fill=(82, 81, 78), font=font)
            for i, (_, row) in enumerate(chunk.iterrows()):
                sheet.paste(_frame_tile(row, masks[row["row"]], font, with_video=True),
                            (GAP + (i % cols) * (W + GAP), TOP + (i // cols) * (H + CAP + GAP)))
            name = f"check_{key}_p{p + 1:02d}.jpg"
            sheet.save(os.path.join(out_dir, name), quality=88)
            check_pages.append(name)
    risky = pd.concat([subset.assign(why=key) for key, (_, subset) in checks.items()])

    table = df.drop(columns=["row", "member"]).assign(coverage_pct=(df["coverage"] * 100).round(1),
                                                      score=df["score"].round(3), bad_mask="")
    table = table[["page", "identity", "sequence", "split", "frame", "label", "score", "coverage_pct", "usable", "bad_mask"]]
    table.to_csv(os.path.join(out_dir, "masks.csv"), index=False)
    return pages, check_pages, risky


def review_preview(path, width=900):
    """A small JPEG copy of a review sheet, for showing inside a notebook without bloating it."""
    import io
    from IPython.display import Image as NotebookImage
    image = Image.open(path)
    image = image.resize((width, int(image.height * width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return NotebookImage(data=buffer.getvalue(), format="jpeg")
