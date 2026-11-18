# Notes

## Swin Transformer (MegaDescriptor's backbone)

MegaDescriptor-S-224, the model `preprocess.py` uses to extract frame features, is built on a **Swin Transformer**. Its full architecture name is `swin_small_patch4_window7_224`: Swin, small size, 4×4 patches, 7×7 windows, 224×224 input.

### Background: a normal Transformer

Transformers come from language models. Text is split into tokens (roughly words), and **attention** lets every token look at _every other token_ to decide what's relevant. This is powerful, but the cost grows with the square of the number of tokens.

### Vision Transformer (ViT)

ViT is the first straightforward way to apply this to images:

- It cuts the image into patches, usually 16×16 pixels. A 224×224 image becomes 14×14 = 196 patches.
- Each patch is treated like a word, and every patch attends to every other patch.
- Weaknesses: the patches are coarse, and smaller patches would blow up the cost (4×4 patches would give 3,136 tokens, about 9.8 million attention pairs per layer). It also works at only one scale throughout.

### Swin ("Shifted windows")

Swin fixes these problems with three ideas:

1. **Small patches, local attention.** It uses 4×4 patches, which keeps finer detail. Attention only happens inside small **windows** of 7×7 patches, not across the whole image, so the cost stays low.
2. **Shifted windows.** If windows never talked to each other, information couldn't spread. So every other layer shifts the window grid by half a window, and patches near a border end up sharing a window with their neighbors. Over several layers, information travels across the whole image.
3. **A hierarchy, like a CNN.** After each stage, neighboring patches are merged (2×2 into 1). The grid shrinks while each patch describes a larger area:

| Stage | Grid size | Each patch covers | Feature size |
| ----- | --------- | ----------------- | ------------ |
| 1     | 56×56     | 4×4 px            | 96           |
| 2     | 28×28     | 8×8 px            | 192          |
| 3     | 14×14     | 16×16 px          | 384          |
| 4     | 7×7       | 32×32 px          | **768**      |

At the end, the 7×7 grid is averaged into a single **768-number vector**. That is the embedding saved for each frame in `polarbear_embeddings.npz`.

### Comparison

|           | Normal Transformer  | ViT                  | Swin                        |
| --------- | ------------------- | -------------------- | --------------------------- |
| Input     | text tokens         | 16×16 image patches  | 4×4 image patches           |
| Attention | all tokens          | all patches (global) | within 7×7 windows, shifted |
| Scale     | one                 | one                  | multi-scale (4 stages)      |
| Cost      | grows quadratically | grows quadratically  | roughly linear              |

### Why it suits polar bear re-ID

The early stages see fine local detail like fur texture, scars and face shape. The later stages see the overall body shape. Both help tell one bear from another.

### Sources

- Model card: https://huggingface.co/BVRA/MegaDescriptor-S-224 (architecture `swin_small_patch4_window7_224`; `config.json` gives `num_features: 768`, `input_size: [3, 224, 224]`)
- Swin Transformer paper: Liu et al., 2021, https://arxiv.org/abs/2103.14030
