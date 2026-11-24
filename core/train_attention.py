import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Learned attention model / from Cindy's code

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(self, video_features):
        Q = self.query(video_features)
        K = self.key(video_features)
        V = self.value(video_features)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_features = torch.bmm(attention_weights, V)

        return attended_features, attention_weights

class VideoReIDModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(VideoReIDModel, self).__init__()
        self.temporal_attention = TemporalAttention(feature_dim)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, video_features):
        attended_features, attention_weights = self.temporal_attention(video_features)

        video_representation = attended_features.mean(dim=1)
        
        output = self.fc(video_representation)
        
        return output, video_representation

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

def train_model(video_features, labels, video_df, num_epochs=40, batch_size=64, seed=0):
    torch.manual_seed(seed)

    train_idx = torch.tensor(video_df.index[video_df["split"] == "train"].values)

    # Initialize model, loss function, and optimizer
    model = VideoReIDModel(feature_dim=video_features.shape[-1], num_classes=len(labels.unique()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        shuffled_idx = train_idx[torch.randperm(len(train_idx))]

        for batch_idx in shuffled_idx.split(batch_size):
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(video_features[batch_idx])
            loss = criterion(outputs, labels[batch_idx])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model
