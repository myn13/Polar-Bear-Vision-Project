import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def aggregate_video_embeddings(npz_path="polarbear_embeddings.npz"):
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
        mean_emb = np.mean(group_embs, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        seq_embeddings.append(mean_emb)
        seq_metadata.append({
            "identity": identity,
            "sequence": sequence,
            "split": split
        })
    return np.array(seq_embeddings), pd.DataFrame(seq_metadata)

def calculate_rank1_accuracy(seq_embeddings, seq_df):
    train_idx = seq_df.index[seq_df["split"] == "train"].tolist()
    test_idx = seq_df.index[seq_df["split"] == "test"].tolist()
    gallery_embs = seq_embeddings[train_idx]
    gallery_ids = seq_df.iloc[train_idx]["identity"].values
    query_embs = seq_embeddings[test_idx]
    query_ids = seq_df.iloc[test_idx]["identity"].values
    sim_matrix = cosine_similarity(query_embs, gallery_embs)
    
    correct_matches = 0
    for i in range(len(query_embs)):
        best_match_idx = np.argmax(sim_matrix[i])
        predicted_id = gallery_ids[best_match_idx]
        
        if predicted_id == query_ids[i]:
            correct_matches += 1
            
    rank1_acc = (correct_matches / len(query_embs)) * 100
    print(f"Polar Bear Rank-1 Accuracy: {rank1_acc:.2f}%")
    
    return rank1_acc

def plot_comparison_results(polar_bear_acc):
    data = {
        "Dataset": ["PolarBearVidID (Ours)", "AAUZebraFish", "ATRW", "Cows2021", 
                    "MacaqueFaces", "BirdIndividualID", "SeaTurtleIDHeads", "BelugaID"],
        "Rank-1 Accuracy (%)": [polar_bear_acc, 99.93, 94.33, 99.54, 99.04, 97.82, 91.18, 66.48]
    }
    df_comparison = pd.DataFrame(data)
    df_comparison = df_comparison.sort_values(by="Rank-1 Accuracy (%)", ascending=True)
    df_comparison.to_csv("core_comparison_table.csv", index=False)
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4" if ds != "PolarBearVidID (Ours)" else "#d62728" for ds in df_comparison["Dataset"]]
    bars = plt.barh(df_comparison["Dataset"], df_comparison["Rank-1 Accuracy (%)"], color=colors)
    plt.xlabel("Rank-1 Accuracy (%)", fontsize=12)
    plt.title("MegaDescriptor-S-224 Performance: Polar Bear vs Wildlife Datasets", fontsize=14)
    plt.xlim(0, 110)
    
    for bar in bars:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.2f}%', va='center', fontsize=10)
                 
    plt.tight_layout()
    plt.savefig("core_comparison_chart.png", dpi=300)
    plt.show()