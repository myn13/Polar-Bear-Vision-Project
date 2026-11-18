# Polar-Bear-Vision-Project
Feature vectors for each frame are stored here: https://drive.google.com/file/d/1ZQj7SaLEsWEthNlaP6GhSHxAR2eLffKo/view?usp=sharing
Note that for each video sequence, 20 frames are selected at a roughly equal spacing. 

## Project layout

```
preprocess.py, core_objective.py, learned_attention.py, logging_utils.py   code
notebooks/   Jupyter notebooks
data/        generated data: splits/ (train/validation/test CSVs) and polarbear_embeddings.npz
results/     outputs for the report: comparison table and chart
docs/        notes and TODO list
```

Put the PolarBearVidID dataset folder in the repo folder (it is gitignored). Notebooks in `notebooks/` start with a setup cell that switches to the repo folder, so imports and paths like `data/splits` work. Copy that cell into any new notebook.

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
