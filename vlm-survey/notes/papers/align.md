# ALIGN (2102.05918v2) -- extraction for ch-shared-space delta

## Mechanism
Same dual-encoder idea as CLIP with a different data bet: minimal frequency-based filtering of raw alt-text instead of curation.
Image tower EfficientNet (global pooling, no 1x1 conv head); text tower BERT with [CLS] embedding plus a linear FC layer to
match dimension; 100k wordpiece vocab from own data; both trained from scratch. Cosine similarity at the top.
Headline model: EfficientNet-L2 + BERT-Large; image res 289x289 (resize 346x346, random crop).

## Loss (Sec. 3 / "Pre-training"; PDF equations (1) and (2), LaTeX labels eq:i2t_loss, eq:t2i_loss)
"We minimize the sum of two losses" (sum, not average -- CLIP averages).
Eq. (1), image-to-text:
L_{i2t} = -\frac{1}{N}\sum_i^N\log{\frac{\exp(x_i^\top y_i / \sigma)}{\sum_{j=1}^{N} \exp(x_i^\top y_j / \sigma)}}
Eq. (2), text-to-image:
L_{t2i} = -\frac{1}{N}\sum_i^N\log{\frac{\exp(y_i^\top x_i / \sigma)}{\sum_{j=1}^{N} \exp(y_i^\top x_j / \sigma)}}
Gloss: x_i = normalised image embedding of pair i; y_j = normalised text embedding of pair j; N = batch size;
sigma = temperature scaling the logits. Positives = matched pairs; negatives = all other pairs in the batch. Called
"normalized softmax loss" (Zhai and Wu). Embeddings from all cores are concatenated to form a much larger batch.

## Temperature
- Learned: "we find that it can be effectively learned together with all the other parameters" (Sec. 3).
- Init 1.0 (Sec. 5 "The softmax temperature variable is initialized as 1.0"), shared between i2t and t2i losses. Label smoothing 0.1.
- Ablation (Sec. 8.1): learned baseline "converged to about 1/64"; hand-picked fixed values "could be slightly better" but learnable kept;
  temperature drops to ~1.2x of converged value in the first 100k steps.
- Batch: 1024 TPUv3 cores x 16 positive pairs = effective batch 16384. LAMB, 1.2M steps (~12 epochs). Fine-tune retrieval uses 2048.

## Dataset (Sec. 3 Data)
- "much larger (1.8B image-text pairs) but noisier dataset" (Sec. 3 first para). Abstract/intro say "over one billion noisy image alt-text pairs".
  Intro: "two orders of magnitude larger than the Conceptual Captions dataset".
- Filtering (Sec. 3): only "simple frequency-based filtering".
  Image-based: remove pornographic images; keep only images whose shorter dimension > 200 px and aspect ratio < 3;
  discard images with more than 1000 associated alt-texts; remove duplicates/near-duplicates of test images.
  Text-based: exclude alt-texts shared by more than 10 images; discard alt-texts containing any rare token (outside the 100 million
  most frequent unigrams and bigrams of the raw dataset); discard texts shorter than 3 unigrams or longer than 20 unigrams.
- Ablation (Table 8): ALIGN data worse than CC at 3M, but at 6M/12M catches up; "ALIGN data outperforms Conceptual Captions with only 4x size".
  Scale beats noise.

## Zero-shot ImageNet (Table 4)
ALIGN 76.4 vs CLIP 76.2 (ImageNet); ImageNet-R 92.2 vs 88.9; ImageNet-A 75.8 vs 77.2; ImageNet-V2 70.1 vs 70.1.
Classnames fed directly into the text encoder (no prompt template described). Frozen-feature ImageNet 85.5; fine-tuned 88.64.

## Outline assertions I could not confirm or that are wrong
- "1.8B uncurated alt-text pairs": paper says 1.8B pairs after minimal frequency-based filtering (not fully unfiltered;
  "uncurated" fine, "unfiltered" wrong). Abstract only says "over one billion".
- ALIGN vs CLIP loss: not identical -- ALIGN sums i2t and t2i losses and (final) uses label smoothing 0.1 and init temperature 1.0;
  CLIP averages and inits at 0.07 with a clip at logit scale 100 (ALIGN mentions no clip).
- ALIGN encoders are EfficientNet + BERT (CNN + transformer), not ViT; CLIP's headline is ViT-L/14@336.
