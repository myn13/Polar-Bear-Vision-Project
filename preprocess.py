import numpy as np
import os
import re
import pandas as pd
from pathlib import Path
from PIL import Image
import timm
from torchvision import transforms as T
from tqdm.auto import tqdm
import torch
from sklearn.model_selection import train_test_split
from logging_utils import get_logger

logger = get_logger(__name__)


def parse_file(filepath):
    path = Path(filepath)
    identity = path.parent.name
    match = re.fullmatch(r"(?P<sequence>[0-9]{3}C[0-9]+T[0-9]+)F(?P<frame>[0-9]+)", path.stem)
    if not match: return None
    return {
        "filepath": str(path),
        "identity": identity if re.fullmatch(r"[0-9]{3}", identity) else match.group("sequence")[:3],
        "sequence": match.group("sequence"), 
        "frame_number": int(match.group("frame"))
    }


def log_frame_counts(df, frames_per_video):
    """Log how many frames each video has, and warn about videos shorter than frames_per_video."""
    frame_counts = df.groupby(["identity", "sequence"]).size()
    logger.info("%d videos | frames per video: min %d, max %d",
                len(frame_counts), frame_counts.min(), frame_counts.max())
    short_videos = frame_counts[frame_counts < frames_per_video]
    if not short_videos.empty:
        logger.warning("%d videos have fewer than %d frames, keeping all their frames: %s",
                       len(short_videos), frames_per_video, short_videos.to_dict())


def build_manifest_and_sample(records, frames_per_video = 20):
    """
    Pick 20 evenly spaced frames from each video sequence

    Input : 
    records is a list of dictionaries built in run_pipeline()
    eg. [{filepath : file.jpg, identity, sequence, frame, member, split}]
    """
    df = pd.DataFrame(records)
    log_frame_counts(df, frames_per_video)
    sampled_groups = []
    for _, video in df.groupby(["identity", "sequence"]): 
        video = video.sort_values("frame_number")
        n = min(frames_per_video, len(video))
        indices = np.linspace(0, len(video) - 1, n, dtype=int) 
        sampled_groups.append(video.iloc[indices]) 
    sampled_df = pd.concat(sampled_groups, ignore_index=True)   # lists the frames ordered by: bear, video sequence, frame number 
    return sampled_df


def split_video_dataset(sampled_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split video sequences 70/15/15 into train/val/test, so all frames of a video stay in one split (prevents leakage)
    Stratified by bear identity, so every bear has about the same share in each split to ensure class 
    
    1431 video sequences is split into:
    1001 train, 215 validation and 215 test set

    Takes the table of sampled frames, 28,620 rows with one per frame and no split yet, 
    and returns the same 28,620 frames with a split column that puts each video's 20 frames 
    together in train, validation or test.
    """
    videos = sampled_df[["identity", "sequence"]].drop_duplicates().reset_index(drop=True) # 1431 video sequences in total from 13 polar bears 
    train_vids, remaining = train_test_split(
        videos, 
        test_size=(val_ratio + test_ratio), 
        stratify=videos["identity"],    # ensure that every bear is represented fairly in every split
        random_state=random_seed
    )
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_vids, test_vids = train_test_split(
        remaining, 
        test_size=val_test_ratio, 
        stratify=remaining["identity"], 
        random_state=random_seed
    )
    splits = {"train": train_vids, "validation": val_vids, "test": test_vids}
    split_labels = pd.concat(
        [df.assign(split=name) for name, df in splits.items()], 
        ignore_index=True
    )
    manifest = sampled_df.merge(split_labels, on=["identity", "sequence"], how="left")
    return manifest


def save_splits_to_csv(manifest, output_dir="data/splits"):
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        split_df = manifest[manifest["split"] == split_name]
        file_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(file_path, index=False)
        logger.info("Saved %d frames to %s", len(split_df), file_path)


def load_feature_extractor(device="cpu"): 
    """
    Load the pretrained MegaDescriptor model
    Reference: https://huggingface.co/BVRA/MegaDescriptor-S-224
    """
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-S-224", pretrained=True) 
    model = model.to(device).eval() # evaluation mode: In Swin transformer, during training it has some dropouts to make training more robust. Evaluation turns it off, so the same image always gives the same embedding.
    model.requires_grad_(False) # freeze the model so none of its weights get updated
    transform = T.Compose([ 
        T.Resize((224, 224)),   # Resize image to 224x224 with each pixel values between 0-255 -- TODO the dataset images are 256x128 (2:1 aspect ratio) so every bear gets squahsed since this resizing doesnt keep aspect ratio
        T.ToTensor(),   # convert to PyTorch tensor - Expected shape: (3, 224, 224)
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # Expected shape: (3, 224, 224) with each pixel values between -1 to 1 -- Matches example on MegaDescriptor model card
    ]) 
    return model, transform


def extract_and_save_embeddings(manifest_df, model, transform, output_path, batch_size=32, device="cpu"):
    """
    Runs every sampled frame through MegaDescriptor and saves the embedding or fingerprints in one file
    Takes 28620 rows, model transforms each to a 768 number embedding per frame
    """
    parts = []
    for start in tqdm(range(0, len(manifest_df), batch_size), desc="Extracting embeddings"):
        batch_paths = manifest_df["member"].iloc[start : start + batch_size] 
        images = []
        for p in batch_paths:
            with Image.open(p) as img:
                images.append(transform(img.convert("RGB")))
        batch_tensor = torch.stack(images).to(device)
        with torch.inference_mode():
            embeddings = model(batch_tensor).cpu()
            parts.append(embeddings)
    all_embeddings = torch.cat(parts).numpy()
    np.savez_compressed(
        output_path,
        embeddings=all_embeddings,
        **{col: manifest_df[col].to_numpy(dtype=str) 
           for col in ["member", "identity", "sequence", "split"]}
    )
    logger.info("Saved embeddings to %s with shape %s", output_path, all_embeddings.shape)


def run_pipeline(root_dir, output_npz="data/polarbear_embeddings.npz", csv_out_dir="data/splits"):
    records = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            filepath = os.path.join(root, f)
            info = parse_file(f) 
            if info:
                info["member"] = filepath  
                records.append(info)
    sampled_df = build_manifest_and_sample(records, frames_per_video=20)
    manifest = split_video_dataset(sampled_df)
    save_splits_to_csv(manifest, output_dir=csv_out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_feature_extractor(device=device)
    extract_and_save_embeddings(
        manifest_df=manifest, 
        model=model, 
        transform=transform, 
        output_path=output_npz, 
        batch_size=32, 
        device=device
    )

