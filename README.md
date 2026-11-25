# Polar Bear Vision Project

CS 117 (Computer Vision), Team 9.

We re-identify **individual polar bears from short video clips**: given a clip of a bear, say which of the 13 known bears it is. Each frame is turned into an embedding by a pretrained wildlife re-identification model (**MegaDescriptor-S-224**), the frames of each video are combined into one video embedding, and a test video is identified by finding the most similar training video.

The research question is whether **learning how to combine the frames** (a temporal attention model) beats simply averaging them.

## Contributions
| Name | Contribution  | 
|---|---|
| Shizuka Takao | Built the initial MegaDescriptor Colab prototype, improved pipeline organization and documentation, extended baseline comparisons across 29 datasets, and developing attention heatmaps for failure analysis | 
| Cindy Zhang | Implemented learned attention layer and combined learned attention with MegaDescriptor to create video identification model. Graphed results of learned attention model versus MegaDescriptor for analysis. | 
|  |  | 
|  |  | 


## Setup

1. **Python environment.** The pipeline was last run with Python 3.13.

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Dataset (only needed to rerun preprocessing).** Download `PolarBearVidID.zip` (1.6 GB) from Zenodo (link below) and unzip it into `data/PolarBearVidID/`, so the images are at `data/PolarBearVidID/000/000C01T000F000.jpg` and so on. The zip has no top-level folder (it contains `000/`, `001/`, ... directly), so create `data/PolarBearVidID/` first and unzip into it.

## Data

- **Dataset:** PolarBearVidID on Zenodo: https://zenodo.org/records/7564529
  - Dataset paper: Zuerl et al., "PolarBearVidID: A Video-based Re-Identification Benchmark Dataset for Polar Bears", *Animals* 13(5), 2023. https://doi.org/10.3390/ani13050801
- **Frame embeddings** polarbear_embeddings.npz
- **Model weights:** MegaDescriptor-S-224 on Hugging Face: https://huggingface.co/BVRA/MegaDescriptor-S-224
- **Trained learned attention weights** (5 seeds: `temporal_attention_seed0.pt` to `temporal_attention_seed4.pt`) on Google Drive: https://drive.google.com/drive/folders/1jL_ygEsEwQ_Scr9zLEIoHD-070TPRBFM?usp=sharing. Put them in `models/`, then run `python -m core.evaluate_attention`.

## How to run

| Goal | What to run | Needs dataset? | Time |
|---|---|---|---|
| Explore the data (counts, splits, leakage check, embeddings) | `notebooks/data_understanding.ipynb` | no | seconds |
| Run the baseline and make the comparison chart | section 2 of `notebooks/preprocess_and_make_comparison.ipynb` | no | seconds |
| Regenerate the splits and embeddings | section 1 of `notebooks/preprocess_and_make_comparison.ipynb` | yes | about 24 min on a laptop CPU, faster with a GPU |
| Train the learned attention model (saves `models/temporal_attention_seed<N>.pt`) | `python -m core.train_attention --seed 0` (one run per seed; add `--device cuda` for a GPU) | no | about 1 min per seed on a laptop CPU |
| Compare mean pooling and learned attention on validation | `python -m core.evaluate_attention` (every `models/temporal_attention_seed*.pt`) | no | seconds |

Run the commands from the repo folder. If `data/splits/*.csv` and `data/polarbear_embeddings.npz` already exist, `run_pipeline` skips preprocessing entirely, so **Run All** works without the dataset. To regenerate them, put the dataset in `data/PolarBearVidID/` and call `preprocess.run_pipeline(..., overwrite=True)`.

### Project layout

```
Polar-Bear-Vision-Project/
├── README.md
├── requirements.txt
├── core/                       main pipeline
│   ├── preprocess.py           frames → splits + embeddings
│   ├── core_objective.py       baseline: video embeddings, Rank-1, comparison chart
│   ├── learned_attention.py    learned attention model (self-attention over the frames)
│   ├── train_attention.py      trains the learned attention model, saves its weights
│   ├── evaluate_attention.py   mean pooling vs learned attention on validation
│   └── logging_utils.py        colored console logger used by the scripts
├── experiments/                analyses of the baseline
│   ├── heatmaps.py             which parts of each frame made a match
│   ├── background_test.py      bear vs background: does the model use the enclosure?
│   ├── layerwise.py            how much identity each backbone stage carries
│   ├── two_bears.py            does a second bear in the crop cause mix-ups?
│   └── zoo_analysis.py         do the mistakes land on zoo-mates?
├── notebooks/
│   ├── preprocess_and_make_comparison.ipynb   runs preprocess + baseline
│   ├── data_understanding.ipynb               explores the splits and embeddings
│   ├── experiments/                           one notebook per analysis
│   └── archive/cs_117_project.ipynb           earlier self-contained Colab version
├── data/
│   ├── PolarBearVidID/     the dataset (you download it; gitignored)
│   ├── splits/             train.csv, validation.csv, test.csv (one row per sampled frame)
│   └── polarbear_embeddings.npz   28,620 × 768 embeddings + labels
├── models/                 trained learned attention weights (gitignored)
├── results/                comparison tables and charts
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

`learned_attention.py` (`VideoReIDModel`): a self-attention layer over the 20 frame embeddings of a video (every frame attends to every other frame), a learned layer, then the average over the frames, followed by a classifier over the 13 bears. `train_attention.py` trains it on the training videos only. `evaluate_attention.py` scores it like the baseline: cosine nearest neighbour on the video embedding, validation videos as queries, training videos as the gallery.

## Timeline

| Task | Assigned to | Due | Status |
|---|---|---|---|
| Preprocess data | My, Shizuka | Nov 3 | Done |
| Run model on polar bear data and graph results | Kavita, My | Nov 10 | Done |
| Initial group analysis | Everyone | Nov 15 | Done |
| Implement learned attention model | Cindy, Shizuka, Kavita | Nov 15 | Done |

### After Nov 17

| Task | Assigned to | Status |
|---|---|---|
| Preprocess and split data | Kavita, Cindy | Done|
| Train model | Shizuka | Done |
| Test and graph results | Kavita, Cindy | To do |
| Group analysis | Everyone | To do |


## References
- **Model paper:** Čermák et al., "WildlifeDatasets: An open-source toolkit for animal re-identification", WACV 2024. https://arxiv.org/abs/2311.09118 ([PDF](https://arxiv.org/pdf/2311.09118); results by model size are in Table 5)
- **Backbone paper:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021. https://arxiv.org/abs/2103.14030
