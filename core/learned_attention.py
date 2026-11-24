import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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