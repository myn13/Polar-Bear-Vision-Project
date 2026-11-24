"""
Extension result: mean pooling vs the learned attention model (Cindy's VideoReIDModel), on the
validation videos.

Loads the weights that train_attention.py saved and scores the model the same way as the baseline in
core_objective.py: validation videos are the queries, training videos the gallery, cosine nearest
neighbour on the video embedding. The test split is never used.

A trained model depends on its random seed, so pass one weights file per seed: the summary gives the
average and the lowest/highest over the runs, and a second file keeps every run's score. Mean pooling
is not trained, so it has one score.

Run from the repo folder:
    python -m core.evaluate_attention                                  (every models/temporal_attention_seed*.pt)
    python -m core.evaluate_attention --weights models/temporal_attention_seed0.pt
"""

import argparse
import glob
import os

import pandas as pd
import torch

from core.core_objective import aggregate_video_embeddings, calculate_rank1_accuracy
from core.learned_attention import VideoReIDModel
from core.train_attention import load_video_features, score_on_validation


def load_model(weights, feature_dim, num_classes, device="cpu"):
    """Cindy's model with the parameters that train_attention.py saved."""
    model = VideoReIDModel(feature_dim=feature_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device)


def compare(embeddings_path, weights_paths, device="cpu"):
    """
    Rank-1 on validation: mean pooling, and the learned attention model once per weights file (one per
    seed). Returns (summary, runs): summary has one row per method, with the learned model's average and
    lowest/highest over its runs; runs has one row per weights file.
    """
    mean_embs, mean_df = aggregate_video_embeddings(embeddings_path)
    n = int((mean_df["split"] == "validation").sum())
    mean_acc = calculate_rank1_accuracy(mean_embs, mean_df, query_split="validation")

    video_features, labels, video_df = load_video_features(embeddings_path)
    rows = []
    for weights in weights_paths:
        model = load_model(weights, video_features.shape[-1], len(labels.unique()), device)
        rows.append({"weights": weights, "rank1": score_on_validation(model, video_features, video_df, device)})
    runs = pd.DataFrame(rows)

    accs = runs["rank1"]
    summary = pd.DataFrame([
        {"method": "Mean pooling", "runs": 1, "rank1_pct": round(mean_acc, 2),
         "lowest": round(mean_acc, 2), "highest": round(mean_acc, 2), "videos": n},
        {"method": "Learned attention", "runs": len(accs), "rank1_pct": round(accs.mean(), 2),
         "lowest": round(accs.min(), 2), "highest": round(accs.max(), 2), "videos": n},
    ])
    runs = runs.assign(rank1_pct=accs.round(2), correct=(accs * n / 100).round().astype(int),
                       videos=n).drop(columns="rank1")
    return summary, runs


def main():
    parser = argparse.ArgumentParser(
        description="Compare mean pooling and the trained learned attention model on the validation videos "
                    "(the test split is never used). Run from the repo folder: python -m core.evaluate_attention")
    parser.add_argument("--embeddings", default="data/polarbear_embeddings.npz", help="frame embeddings (default: %(default)s)")
    parser.add_argument("--weights", nargs="+", default=["models/temporal_attention_seed*.pt"],
                        help="one or more trained weights files, e.g. one per seed; a quoted pattern like "
                             '"models/temporal_attention_seed*.pt" works in any shell (default: %(default)s)')
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda" (default: %(default)s)')
    parser.add_argument("--out", default="results/attention_comparison.csv",
                        help="where to save the summary table; every run's score goes next to it in *_runs.csv "
                             "(default: %(default)s)")
    args = parser.parse_args()

    weights = []
    for w in args.weights:
        found = sorted(glob.glob(w)) if any(c in w for c in "*?[") else [w]
        missing = [p for p in found if not os.path.exists(p)]
        if not found or missing:
            parser.error(f"no weights at {w}; train first with: python -m core.train_attention")
        weights += [p for p in found if p not in weights]

    summary, runs = compare(args.embeddings, weights, args.device)
    print("\nEvery run:\n" + runs.to_string(index=False))
    print("\nSummary:\n" + summary.to_string(index=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    runs_out = "{}_runs{}".format(*os.path.splitext(args.out))
    summary.to_csv(args.out, index=False)
    runs.to_csv(runs_out, index=False)
    print(f"\nSaved {args.out} and {runs_out}")


if __name__ == "__main__":
    main()
