"""
Steps
1. Loads frame fingerperints
2. Averages each video's 20 frames into one video fingerprint
3. matches every validation video to its most similar training video (the test split stays held out)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from core.logging_utils import get_logger

logger = get_logger(__name__)

def aggregate_video_embeddings(npz_path="data/polarbear_embeddings.npz"):
    data = np.load(npz_path)
    df = pd.DataFrame({
        "identity": data["identity"],
        "sequence": data["sequence"],
        "split": data["split"]
    })
    embeddings = data["embeddings"]
    seq_embeddings = []
    seq_metadata = []

    for (identity, sequence, split), group_indices in df.groupby(["identity", "sequence", "split"]).groups.items():
        group_embs = embeddings[group_indices] 
        mean_emb = np.mean(group_embs, axis=0)  # averages the 20 fingerprints into one. Expected shape : (768,)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)  # np.linarlg.norm measures how long a vector is. Dividing by it makes the length of the vector 1 in same direction
        seq_embeddings.append(mean_emb)
        seq_metadata.append({
            "identity": identity,
            "sequence": sequence,
            "split": split
        })
    return np.array(seq_embeddings), pd.DataFrame(seq_metadata)

def calculate_rank1_accuracy(seq_embeddings, seq_df, query_split="validation"):
    """
    For each query video (validation by default; the test split stays held out),
    it finds the most similar training video and checks whether that video shows the same bear
    """
    train_idx = seq_df.index[seq_df["split"] == "train"].tolist()
    query_idx = seq_df.index[seq_df["split"] == query_split].tolist()
    gallery_embs = seq_embeddings[train_idx]
    gallery_ids = seq_df.iloc[train_idx]["identity"].values
    query_embs = seq_embeddings[query_idx]
    query_ids = seq_df.iloc[query_idx]["identity"].values
    sim_matrix = cosine_similarity(query_embs, gallery_embs)
    
    correct_matches = 0
    for i in range(len(query_embs)):
        best_match_idx = np.argmax(sim_matrix[i])
        predicted_id = gallery_ids[best_match_idx]
        
        if predicted_id == query_ids[i]:
            correct_matches += 1
            
    rank1_acc = (correct_matches / len(query_embs)) * 100
    logger.info("Polar Bear Rank-1 Accuracy: %.2f%% (%d/%d %s videos)",
                rank1_acc, correct_matches, len(query_embs), query_split)
    
    return rank1_acc

# MegaDescriptor-S-224 accuracy (%) on other wildlife datasets.
# Source: Čermák et al., WACV 2024, Table 5 (https://arxiv.org/pdf/2311.09118), S-224 column.
MEGADESCRIPTOR_S224_ACC = {
    "AAUZebraFish": 99.55, "ATRW": 93.40, "AerialCattle2017": 100.0, "BelugaID": 38.84,
    "BirdIndividualID": 96.81, "CTai": 87.46, "CZoo": 98.11, "Cows2021": 98.73,
    "FriesianCattle2015": 55.00, "FriesianCattle2017": 96.46, "GiraffeZebraID": 68.40,
    "Giraffes": 80.97, "HappyWhale": 14.98, "HumpbackWhaleID": 36.25, "HyenaID2022": 66.67,
    "IPanda50": 79.85, "LeopardID2022": 67.27, "LionData": 16.13, "MacaqueFaces": 98.89,
    "NDD20": 58.57, "NOAARightWhale": 24.95, "NyalaData": 14.95, "OpenCows2020": 99.58,
    "SMALST": 100.0, "SeaTurtleIDHeads": 83.74, "SealID": 63.31, "StripeSpotter": 94.51,
    "WhaleSharkID": 32.74, "ZindiTurtleRecall": 38.38,
}
OURS = "PolarBearVidID (ours)"


def plot_comparison_results(polar_bear_acc):
    """Compare our Rank-1 accuracy with MegaDescriptor-S-224's reported accuracy on 29 other datasets."""
    df_comparison = pd.DataFrame(
        list(MEGADESCRIPTOR_S224_ACC.items()) + [(OURS, polar_bear_acc)],
        columns=["Dataset", "Rank-1 Accuracy (%)"],
    ).sort_values("Rank-1 Accuracy (%)").reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    df_comparison.to_csv("results/core_comparison_table.csv", index=False)

    rank = (df_comparison["Rank-1 Accuracy (%)"] > polar_bear_acc).sum() + 1
    logger.info("Polar bears rank %d of %d datasets", rank, len(df_comparison))

    # Colors: other datasets recede, ours stands out
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    others, ours = "#c3c2b7", "#2a78d6"
    is_ours = df_comparison["Dataset"] == OURS

    fig, ax = plt.subplots(figsize=(9, 10), facecolor=surface)
    ax.set_facecolor(surface)
    ax.barh(df_comparison["Dataset"], df_comparison["Rank-1 Accuracy (%)"],
            color=[ours if o else others for o in is_ours], height=0.7)

    # Dashed line at our accuracy: datasets above/below it are easy to see
    ax.axvline(polar_bear_acc, color=ours, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    y_ours = df_comparison.index[is_ours][0]
    ax.text(polar_bear_acc + 1.5, y_ours, f"{polar_bear_acc:.2f}%", va="center",
            fontsize=11, fontweight="bold", color=ink)

    # Axis styling: light grid, no box
    ax.set_xlim(0, 105)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.xaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(axis="both", colors=muted, length=0, labelsize=10)
    for label in ax.get_yticklabels():
        if label.get_text() == OURS:
            label.set_color(ink)
            label.set_fontweight("bold")
    ax.margins(y=0.01)

    # Title, subtitle, and caveat
    fig.text(0.02, 0.985, "Polar bear re-identification vs. other wildlife datasets",
             ha="left", va="top", fontsize=15, fontweight="bold", color=ink)
    fig.text(0.02, 0.957, f"MegaDescriptor-S-224, Rank-1 accuracy. Polar bears rank {rank} of {len(df_comparison)}.",
             ha="left", va="top", fontsize=11, color=muted)
    fig.text(0.02, 0.01,
             "Other datasets: per-image accuracy reported by Čermák et al. (WACV 2024, Table 5); their individuals were likely seen in training.\n"
             "Polar bears (ours): per-video validation accuracy (20 frames averaged) on bears MegaDescriptor never saw.",
             fontsize=8.5, color=muted, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig("results/core_comparison_chart.png", dpi=300, facecolor=surface)
    plt.show()