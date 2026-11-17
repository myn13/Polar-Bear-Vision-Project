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


def build_manifest_and_sample(records, frames_per_video = 20):
    df = pd.DataFrame(records)
    sampled_groups = []
    for _, video in df.groupby(["identity", "sequence"]): 
        video = video.sort_values("frame_number")
        n = min(frames_per_video, len(video))
        indices = np.linspace(0, len(video) - 1, n, dtype=int) 
        sampled_groups.append(video.iloc[indices])
    sampled_df = pd.concat(sampled_groups, ignore_index=True) 
    return sampled_df

def split_video_dataset(sampled_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    videos = sampled_df[["identity", "sequence"]].drop_duplicates().reset_index(drop=True)
    train_vids, remaining = train_test_split(
        videos, 
        test_size=(val_ratio + test_ratio), 
        stratify=videos["identity"], 
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

def save_splits_to_csv(manifest, output_dir="polarbear_splits"):
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        split_df = manifest[manifest["split"] == split_name]
        file_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(file_path, index=False)
        print(f"Save {len(split_df)} into {file_path}")

def load_feature_extractor(device="cpu"): 
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-S-224", pretrained=True) 
    model = model.to(device).eval() 
    model.requires_grad_(False)
    transform = T.Compose([ 
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ]) 
    return model, transform

def extract_and_save_embeddings(manifest_df, model, transform, output_path, batch_size=32, device="cpu"):
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
    print(f"Save successfully {output_path} with shape {all_embeddings.shape}")

def run_pipeline(root_dir, output_npz="polarbear_embeddings.npz", csv_out_dir="polarbear_splits"):
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

