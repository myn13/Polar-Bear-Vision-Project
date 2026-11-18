# MegaDescriptor

What the model we use is, how it was trained, and why we expect it to work on polar bears it has never seen.

We use **MegaDescriptor-S-224** (`hf-hub:BVRA/MegaDescriptor-S-224`) as a frozen feature extractor in `preprocess.py`. It comes from the WildlifeDatasets paper (Čermák, Picek, Adam, Papafitsoros, WACV 2024).

## What it is

| | |
|---|---|
| Backbone | Swin Transformer, small (`swin_small_patch4_window7_224`), 49.6M parameters |
| Input | 3 × 224 × 224 RGB image, normalized with mean/std 0.5 (model card example) |
| Output | one **768-number embedding** per image |
| Classification head | none (`num_classes: 0` in its `config.json`) |
| Other sizes | Tiny, Small, Base (224 px) and Large (224 and 384 px) |

### A descriptor, not a classifier

- A **classifier** is trained on a fixed list of classes and outputs "which class is this?". It can only answer with classes it was trained on.
- A **descriptor** outputs a **fingerprint** (the embedding). Images of the same individual get similar fingerprints and different individuals get different ones. It never says *who* the animal is; you find that out by **comparing** fingerprints.

That's why MegaDescriptor ships without a classification layer: the individuals it was trained on (turtles, cows, macaques, ...) aren't ours, so their class scores would be useless to us. The fingerprints are what transfer.

## How it was trained

- **Data:** "all publicly available" animal re-identification datasets in the WildlifeDatasets toolkit at the time, about 29–31 datasets (late 2023) covering many species: zebrafish, cattle, sea turtles, whales, big cats, primates, birds, and others.
- **Loss: ArcFace.** ArcFace is a classification loss over the training individuals with an added angular *margin*: the model is pushed so that each individual's images sit in a tight cluster on a sphere, clearly separated from every other individual's cluster. The paper compared ArcFace against triplet loss and found ArcFace better.
- **After training,** the ArcFace classification layer is thrown away and only the backbone (which produces the embedding) is published. The clustering it learned is what makes "nearest fingerprint = same individual" work.
- **Optimization** (stated for the paper's training runs): SGD with momentum 0.9, 100 epochs, cosine learning-rate schedule, batch size 128. The paper doesn't say whether the final released models used exactly these, and doesn't state the initialization or augmentations.

### How the paper evaluates it

Same as our baseline: a **1-nearest-neighbor classifier with cosine similarity** in the embedding space, scored by accuracy. Each dataset is split 80/20 into reference and query images in a **closed-set** way, meaning every query individual also appears in the reference set.

## Were our polar bears in its training data?

**No.**

- The weights were uploaded to Hugging Face on **2023-11-06** and haven't changed since (later commits only edit the README).
- PolarBearVidID was added to the WildlifeDatasets library on **2024-02-28**, after the model was trained ("Added MPDD (dogs), PolarBearVidID (polar bears) and SeaStarReID2023 (sea stars)" in the library's changelog).
- PolarBearVidID isn't in the paper's dataset table (Table 1).

So for MegaDescriptor, our 13 bears are **unseen individuals of an unseen species**.

## Why we expect it to work on unseen bears

1. **It learns what makes individuals different, not who specific individuals are.** ArcFace training on many species forces the model to pick up on identity cues such as markings, scars and shapes, and to ignore things like pose. Those cues are useful for any animal, not just the ones it trained on.
2. **Diversity.** Training on dozens of species makes it less likely to depend on cues specific to one species.
3. **Evidence from the paper:** in a cattle experiment, MegaDescriptor "retains reasonable performance on the cattle datasets even when removing cattle images from training". The paper also reports that it beats general-purpose models like CLIP and DINOv2 by a wide margin at animal re-identification.
4. **Evidence from our own result:** the baseline gets **90.23%** Rank-1 on polar bears without any training on them.

### Caveats

- The paper's evidence for unseen species is limited (essentially the one cattle experiment). Most of its reported numbers are on datasets whose individuals **were seen during training**.
- Our 90.23% may partly come from recognizing **zoo backgrounds** rather than the bears themselves (see `todo.md`).

## Our comparison chart: use the S-224 numbers

Our first comparison chart compared our polar bear result against numbers from the paper's **MegaDescriptor-L-384** column (Table 5). We use **S-224**, which is smaller and uses lower-resolution input, and scores lower on the same datasets. `plot_comparison_results` in `core_objective.py` now uses the **S-224** column for all 29 datasets (polar bears rank 12 of 30). The difference for the datasets in the first chart:

| Dataset | L-384 (first chart) | **S-224 (the model we use)** |
|---|---|---|
| BelugaID | 66.48 | **38.84** |
| SeaTurtleIDHeads | 91.18 | **83.74** |
| ATRW | 94.33 | **93.40** |
| BirdIndividualID | 97.82 | **96.81** |
| MacaqueFaces | 99.04 | **98.89** |
| Cows2021 | 99.54 | **98.73** |
| AAUZebraFish | 99.93 | **99.55** |
| PolarBearVidID (ours) | 90.23 | 90.23 |

Two more differences to state when presenting the comparison:

- The other datasets' individuals were most likely **seen during MegaDescriptor's training**; our bears were not. Our setting is harder.
- Our number is **per video** (20 frames averaged); the paper's numbers are **per image**.

## Sources

- Paper: https://arxiv.org/abs/2311.09118 (Sections 4.2, 5, 6; Tables 1, 4, 5)
- Model card: https://huggingface.co/BVRA/MegaDescriptor-S-224
- Model commit history: https://huggingface.co/BVRA/MegaDescriptor-S-224/commits/main
- WildlifeDatasets library and changelog: https://github.com/WildlifeDatasets/wildlife-datasets
