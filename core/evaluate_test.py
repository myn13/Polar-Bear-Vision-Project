import os
import pandas as pd
import torch

from core.core_objective import aggregate_video_embeddings, calculate_rank1_accuracy
from core.evaluate_attention import load_model
from core.train_attention import load_video_features


def main():
    embeddings_path = "data/polarbear_embeddings.npz"
    val_runs_path = "results/attention_comparison_runs.csv"
    out_path = "results/test_attention_comparison.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_runs = pd.read_csv(val_runs_path)
    best_row = val_runs.loc[val_runs["rank1_pct"].idxmax()]
    best_weights = best_row["weights"]
    print(f"Best validation checkpoint: {best_weights} ({best_row['rank1_pct']:.2f}% val)")

    mean_embs, mean_df = aggregate_video_embeddings(embeddings_path)
    n_test = int((mean_df["split"] == "test").sum())
    mean_acc = calculate_rank1_accuracy(mean_embs, mean_df, query_split="test")

    feats, labels, vid_df = load_video_features(embeddings_path)
    model = load_model(best_weights, feats.shape[-1], len(labels.unique()), device)
    model.eval()
    with torch.no_grad():
        out = model(feats.to(device))
        attn_embs = (out[1] if isinstance(out, tuple) else out).cpu().numpy()
    attn_acc = calculate_rank1_accuracy(attn_embs, vid_df, query_split="test")

    results = pd.DataFrame([
        {"method": "Mean pooling", "weights": "N/A", "rank1_pct": round(mean_acc, 2), "videos": n_test},
        {"method": "Learned attention", "weights": best_weights, "rank1_pct": round(attn_acc, 2), "videos": n_test},
    ])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results.to_csv(out_path, index=False)
    print("\nTest Summary:\n" + results.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()