# Polar Bear Vision Project

CS 117 (Computer Vision), Team 9.

We re-identify **individual polar bears from short video clips**: given a clip of a bear, say which of the 13 known bears it is. Each frame is turned into an embedding by a pretrained wildlife re-identification model (**MegaDescriptor-S-224**), the frames of each video are combined into one video embedding, and a test video is identified by finding the most similar training video.

The research question is whether **learning how to combine the frames** (a temporal attention model) beats simply averaging them.

## Setup

1. **Python environment.** The pipeline was last run with Python 3.13.

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Dataset (only needed to rerun preprocessing).** Download `PolarBearVidID.zip` (1.6 GB) from Zenodo (link below) and unzip it into `data/PolarBearVidID/`, so the images are at `data/PolarBearVidID/000/000C01T000F000.jpg` and so on. The zip has no top-level folder (it contains `000/`, `001/`, ... directly), so create `data/PolarBearVidID/` first and unzip into it. The folder and the zip are gitignored, so they won't be committed.

   You don't need the dataset to explore the data or run the baseline: the split CSVs and embeddings are already in `data/`.

## Links

- **Dataset:** PolarBearVidID on Zenodo: https://zenodo.org/records/7564529
  - Dataset paper: Zuerl et al., "PolarBearVidID: A Video-based Re-Identification Benchmark Dataset for Polar Bears", *Animals* 13(5), 2023. https://doi.org/10.3390/ani13050801
- **Frame embeddings** (copy of `data/polarbear_embeddings.npz`): [Google Drive](https://drive.google.com/file/d/1ZQj7SaLEsWEthNlaP6GhSHxAR2eLffKo/view?usp=sharing)
- **Model paper:** Čermák et al., "WildlifeDatasets: An open-source toolkit for animal re-identification", WACV 2024. https://arxiv.org/abs/2311.09118 ([PDF](https://arxiv.org/pdf/2311.09118); results by model size are in Table 5)
- **Model weights:** MegaDescriptor-S-224 on Hugging Face: https://huggingface.co/BVRA/MegaDescriptor-S-224
- **Backbone paper:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021. https://arxiv.org/abs/2103.14030
- **Our notes:** [docs/megadescriptor.md](docs/megadescriptor.md), [docs/swin-transformer.md](docs/swin-transformer.md), [docs/todo.md](docs/todo.md)

## How to run

| Goal | What to run | Needs dataset? | Time |
|---|---|---|---|
| Explore the data (counts, splits, leakage check, embeddings) | `notebooks/data_understanding.ipynb` | no | seconds |
| Run the baseline and make the comparison chart | section 2 of `notebooks/preprocess_and_make_comparison.ipynb` | no | seconds |
| Regenerate the splits and embeddings | section 1 of `notebooks/preprocess_and_make_comparison.ipynb` | yes | about 24 min on a laptop CPU, faster with a GPU |

Embedding extraction is skipped if `data/polarbear_embeddings.npz` already exists (the split CSVs are still rebuilt, which takes seconds). To regenerate the embeddings, call `preprocess.run_pipeline(..., overwrite=True)`.

**Notebooks in `notebooks/`** start with a setup cell that switches to the repo folder, so `import preprocess` and paths like `data/splits` work. Copy that cell into any new notebook you create there.

**Logging.** The scripts print progress with a colored logger (`logging_utils.py`): green for info, yellow for warnings, red for errors.

### Project layout

```
Polar-Bear-Vision-Project/
├── README.md
├── requirements.txt
├── preprocess.py           frames → splits + embeddings
├── core_objective.py       baseline: video embeddings, Rank-1, comparison chart
├── learned_attention.py    temporal attention model (not trained yet)
├── logging_utils.py        colored console logger used by the scripts
├── notebooks/
│   ├── preprocess_and_make_comparison.ipynb   runs preprocess + baseline
│   ├── data_understanding.ipynb               explores the splits and embeddings
│   └── cs_117_project.ipynb                   earlier self-contained Colab version
├── data/
│   ├── PolarBearVidID/     the dataset (you download it; gitignored)
│   ├── splits/             train.csv, validation.csv, test.csv (one row per sampled frame)
│   └── polarbear_embeddings.npz   28,620 × 768 embeddings + labels
├── results/                comparison table and chart
└── docs/                   notes (MegaDescriptor, Swin Transformer) and TODO list
```

## Method

**Data.** PolarBearVidID has 13 individual polar bears (housed in six zoos), 1,431 video sequences and 138,363 frames, at 12.5 frames per second and at most 100 frames per sequence. Filenames look like `000C01T038F005.jpg`: bear `000`, `C01` (probably camera), `T038` (probably track), frame `F005`. A video sequence is everything before the `F`, e.g. `000C01T038`.

`preprocess.py` (`run_pipeline`):

1. Find every frame and read the bear, video and frame number from its filename.
2. Keep **20 evenly spaced frames** per video (138,363 → 28,620 frames).
3. Split the **videos** 70 / 15 / 15 into train / validation / test (1,001 / 215 / 215 videos), stratified by bear. All frames of a video stay in one split, so the model is never tested on a video it has seen.
4. Save the splits to `data/splits/`.
5. Run every sampled frame through frozen MegaDescriptor-S-224 (Swin-Small, 224 × 224 input) to get a 768-number embedding, and save them to `data/polarbear_embeddings.npz`.

`core_objective.py` (baseline):

6. Average each video's 20 frame embeddings into one video embedding.
7. For each test video, find the most similar training video (cosine similarity) and predict its bear. Report Rank-1 accuracy: the share of test videos whose most similar training video shows the same bear.
8. Plot the result next to MegaDescriptor's reported results on other wildlife datasets.

`learned_attention.py` (next step): a self-attention layer over the 20 frame embeddings of a video, followed by a classifier over the 13 bears, so the model can learn which frames matter more.

## Timeline

| Task | Assigned to | Due | Status |
|---|---|---|---|
| Preprocess data | My, Shizuka | Nov 3 | Done |
| Run model on polar bear data and graph results | Kavita, My | Nov 10 | To do |
| Initial group analysis | Everyone | Nov 15 | To do |
| Implement learned attention model | Cindy, Shizuka, Kavita | Nov 15 | To do |

### After Nov 17

| Task | Assigned to | Status |
|---|---|---|
| Preprocess and split data | Kavita, Cindy | To do |
| Train model | Shizuka | To do |
| Test and graph results | Kavita, Cindy | To do |
| Group analysis | Everyone | To do |
