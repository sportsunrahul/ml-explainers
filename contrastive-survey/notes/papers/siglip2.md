# SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

## Identity
- arXiv 2502.14786, version read: v1 (Thu 20 Feb 2025). Michael Tschannen ... Xiaohua Zhai (14 authors, Google DeepMind).
- Venue: not stated. Checkpoints released (big_vision).

## Key idea
Keep SigLIP's sigmoid loss and add, in one recipe: a LocCa captioning decoder (equal weight), SILC/TIPS self-distillation
and masked prediction in the last 20% of training, online data curation (ACID distillation for small models), and a
NaFlex variant for native aspect ratio. Better zero-shot and much better dense/localization features.

## Config (Sec. 2.1)
WebLI (10 billion images, 12 billion alt-texts, 109 languages; 90% English mix), Adam lr 1e-3, wd 1e-4, batch 32k,
cosine with 20k warmup, 40B examples. Sizes B (86M), L (303M), So400m (400M), g (1B).

## Defining equation
Same sigmoid loss as SigLIP (Sec. 2.2); no new contrastive equation.

## Key number
ViT-B/16, 224 px, 196 tokens: zero-shot ImageNet val 78.2 vs SigLIP 76.2 (Table 1, B/16 224 rows, ImageNet-1k val column).
