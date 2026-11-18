# Things to consider / TODO

## High priority

- [ ] **Fix the comparison table: it uses the wrong model size.** `core_comparison_table.csv` and `plot_comparison_results` in `core_objective.py` compare our result against the paper's MegaDescriptor-**L-384** numbers (the large model), but we use **S-224**, which scores lower on the same datasets (paper, Table 5):

  | Dataset | In our table now (L-384) | S-224 (our model) |
  |---|---|---|
  | BelugaID | 66.48 | 38.84 |
  | SeaTurtleIDHeads | 91.18 | 83.74 |
  | ATRW | 94.33 | 93.40 |
  | BirdIndividualID | 97.82 | 96.81 |
  | MacaqueFaces | 99.04 | 98.89 |
  | Cows2021 | 99.54 | 98.73 |
  | AAUZebraFish | 99.93 | 99.55 |
  | PolarBearVidID (ours) | 90.23 | 90.23 |

  Also state when presenting it: the other datasets' individuals were most likely **seen during MegaDescriptor's training**, but our bears weren't (see `megadescriptor.md`), and our number is **per video** while the paper's are **per image**. Check with the team before changing, since this result was already presented.

- [ ] **Is the model recognizing the bear or the background?** The 13 bears live in six different zoos (Zenodo record), and cameras are very uneven across bears (bears `007`–`012` never appear on camera `03`). Since train and test share each bear's zoo and backgrounds, part of the 90.23% may come from recognizing the location. Ideas:
  - look at which bears get confused with each other, and check whether they share a zoo/camera
  - for bears filmed by more than one camera, test on a camera not seen in training

- [ ] **Confirm the filename format and which bear is in which zoo.** `C` = camera and `T` = track are our guesses; the Zenodo page doesn't say. Check the dataset paper (https://doi.org/10.3390/ani13050801), e.g. via the UCI library. Needed for the background question above.

## Medium priority

- [ ] **Aspect ratio of the input images.** Dataset frames are 256 × 128 (2:1), but `load_feature_extractor` stretches them to 224 × 224, which squashes every bear. Try the alternatives and compare Rank-1 against the current **90.23%**. Only the `transform` changes; it needs the full dataset and a GPU (Colab) to re-extract the embeddings.

  | Method                            | Distortion      | Bear cut off | Wasted pixels |
  | --------------------------------- | --------------- | ------------ | ------------- |
  | Stretch (current)                 | yes, 2:1 squash | no           | none          |
  | Pad to square + resize            | no              | no           | about half    |
  | Resize shorter side + center crop | no              | possibly     | none          |

  `Resize(224)` alone (one number) keeps the aspect ratio but gives 448 × 224, which MegaDescriptor rejects (`fixed_input_size: true` in its config).

- [ ] **Normalize each frame before averaging.** In `aggregate_video_embeddings` (`core_objective.py`), the 20 frame embeddings are averaged first and only the average is normalized. Frame embedding lengths range from about 5.6 to 41.8, so long ones dominate the average and the frames don't count equally. Try normalizing each frame to length 1 first, then compare Rank-1 with the current **90.23%**. One-line change; runs locally in seconds from the saved `.npz` (no GPU needed):

  ```python
  group_embs = group_embs / np.linalg.norm(group_embs, axis=1, keepdims=True)
  ```

- [ ] **Neighboring tracks.** Videos like `T037` and `T038` may have been filmed minutes apart in the same conditions. If one is in train and the other in test, the test is easier than it looks. Check after confirming what `T` means.

- [ ] **Normalization values.** The MegaDescriptor model card example uses mean/std `0.5`, but its `config.json` lists the ImageNet values (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`). We use `0.5`. Could compare both.

## Low priority (code cleanup)

- [ ] `run_pipeline` calls `parse_file(f)` with only the filename, so the `filepath` column is just the filename and the full path lives in `member`. Passing `filepath` would be cleaner, but would change the saved CSV/npz columns. Decided to leave it for now.
- [ ] Pass `num_classes=0` explicitly in `timm.create_model(...)` so it's clear we want embeddings, not class scores (currently comes from the model's `config.json`).
- [ ] In `split_video_dataset`, the loop variable `df` in the list comprehension could be renamed to `vids` for clarity.

## Things to keep in mind

- **57 videos don't start at frame 0.** Their frame numbering starts later (e.g. 14, 17). Sampling still works because it picks by position, not frame number.
