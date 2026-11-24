import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from core.learned_attention import VideoReIDModel  # model from Cindy's code
from core.core_objective import calculate_rank1_accuracy

# Training loop

# Groups MegaDescriptor frame embeddings (from Shizuka's notebook) into one [num_frames, feature_dim] tensor per video

def load_video_features(npz_path="data/polarbear_embeddings.npz"):
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    df = pd.DataFrame({
        "identity": data["identity"],
        "sequence": data["sequence"],
        "split": data["split"],
        "frame": pd.Series(data["member"]).str.extract(r"F(\d+)\.jpg$", expand=False).astype(int)
    })

    video_features = []
    video_df = []
    for (identity, sequence, split), group in df.groupby(["identity", "sequence", "split"]):
        group = group.sort_values("frame")
        video_features.append(embeddings[group.index])  # [num_frames, feature_dim]
        video_df.append({"identity": identity, "sequence": sequence, "split": split})

    video_features = torch.tensor(np.stack(video_features))  # [num_videos, num_frames, feature_dim]
    video_df = pd.DataFrame(video_df)

    identity_to_label = {identity: i for i, identity in enumerate(sorted(video_df["identity"].unique()))}
    labels = torch.tensor(video_df["identity"].map(identity_to_label).values)

    return video_features, labels, video_df


# trains model to predict which bear is in each video

def train_model(video_features, labels, video_df, num_epochs=40, batch_size=64, seed=0, lr=0.001, device="cpu",
                verbose=True):
    torch.manual_seed(seed)

    train_idx = torch.tensor(video_df.index[video_df["split"] == "train"].values)

    # Initialize model, loss function, and optimizer
    model = VideoReIDModel(feature_dim=video_features.shape[-1], num_classes=len(labels.unique())).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        shuffled_idx = train_idx[torch.randperm(len(train_idx))]

        for batch_idx in shuffled_idx.split(batch_size):
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(video_features[batch_idx].to(device))
            loss = criterion(outputs, labels[batch_idx].to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def score_on_validation(model, video_features, video_df, device="cpu"):
    """
    Rank-1 the same way as the baseline: cosine nearest neighbour on the video embedding, validation
    videos as queries, training videos as the gallery. The test split is never used.
    """
    keep = video_df["split"].isin(["train", "validation"]).to_numpy(copy=True)  # copy: pandas returns a read-only array
    model.eval()
    with torch.no_grad():
        _, video_embeddings = model(video_features[keep].to(device))
    return calculate_rank1_accuracy(video_embeddings.cpu().numpy(), video_df[keep].reset_index(drop=True),
                                    query_split="validation")


def main():
    parser = argparse.ArgumentParser(
        description="Train the learned attention model on the training videos, then score it on validation "
                    "(validation videos are the queries, training videos the gallery; the test split is never used). "
                    "Run from the repo folder: python -m core.train_attention")
    parser.add_argument("--embeddings", default="data/polarbear_embeddings.npz", help="frame embeddings (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=40, help="default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=64, help="default: %(default)s")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="default: %(default)s")
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda" (default: %(default)s)')
    parser.add_argument("--out", default=None, help="where to save the weights (default: models/temporal_attention_seed<seed>.pt)")
    args = parser.parse_args()
    if args.out is None:
        args.out = f"models/temporal_attention_seed{args.seed}.pt"

    video_features, labels, video_df = load_video_features(args.embeddings)
    model = train_model(video_features, labels, video_df, num_epochs=args.epochs, batch_size=args.batch_size,
                        seed=args.seed, lr=args.lr, device=args.device)

    score_on_validation(model, video_features, video_df, device=args.device)

    # Load later with: model = VideoReIDModel(768, 13); model.load_state_dict(torch.load(path))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    main()
