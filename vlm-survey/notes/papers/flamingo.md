# Flamingo (arXiv 2204.14198v2) -- extraction for ch-flamingo

Source: sources/flamingo/src/*.tex (content.tex = main text; appendix_sections/A_method.tex, B_experiments.tex;
tables/*.tex). Pseudocode lives in figure PDFs (figures/fig3_resampler.pdf, fig4_xattn_dense.pdf), read with pdftotext.
Numbering below follows paper.pdf (main Sec. 2-3; App. A = method, App. B = experiments).

## 1. Gated XATTN-DENSE (Sec. 2.2, Fig. 4 pseudocode, App. A.1.2)

No numbered equation. Verbatim pseudocode from Fig. 4:

```
def gated_xattn_dense(
    y,           # input language features
    x,           # input visual features
    alpha_xattn, # xattn gating parameter - init at 0.
    alpha_dense, # ffw gating parameter - init at 0.
):
    """Applies a GATED XATTN-DENSE layer."""
    # 1. Gated Cross Attention
    y = y + tanh(alpha_xattn) * attention(q=y, kv=x)
    # 2. Gated Feed Forward (dense) Layer
    y = y + tanh(alpha_dense) * ffw(y)
    # Regular self-attention + FFW on language
    y = y + frozen_attention(q=y, kv=y)
    y = y + frozen_ffw(y)
    return y  # output visually informed language features
```

Gloss:
- y = the language hidden states entering the block (queries); x = the 64 visual tokens per image from the Perceiver Resampler (keys and values).
- TWO separate learnable scalars per inserted block: alpha_xattn (gates the cross-attention output) and alpha_dense (gates the new dense FFW output). Both init at 0, so tanh(0) = 0 and both new branches contribute nothing at init.
- tanh(alpha) multiplies the sublayer OUTPUT before it is added to the residual stream (y + tanh(alpha) * sublayer(y)). It does not scale the input.
- Order inside a block: new gated xattn, new gated dense FFW, THEN the frozen LM's own self-attention and FFW (Fig. 4: new block sits before the frozen LM layer; "we insert gated cross attention dense blocks between the original layers").
- Main text (Sec. 2.2): "multiplies the output of a newly added layer by tanh(alpha) before adding it to the input representation from the residual connection, where alpha is a layer-specific learnable scalar initialized to 0" (cites ReZero). "at initialization, the model output matches that of the pretrained LM".
- Squared ReLU is the activation in the new blocks (Table 4); frozen LM uses GeLU.
- Gate evolution (App. A.1.2, Fig. 6, Flamingo-3B, 24 layers): |tanh| gates on both attention and FFW quickly grow from 0 in all layers; values seem to grow with depth, but paper says strong conclusions cannot be drawn (activation scale before gating may vary with depth).
- Ablation (Table 3 iii): removing tanh gating: overall 70.7 -> 66.5 (COCO 86.5 -> 78.4, VATEX 53.4 -> 47.5).

### How often layers are inserted (Table 5, App. B.1.1)
- Flamingo-3B: before EVERY LM block (24 xattn-dense layers, table says 1.2B params; text says 1.4B).
- Flamingo-9B: from the very first layer, before every 4th block (10 layers over 40 LM layers; table 1.6B, text 1.8B).
- Flamingo-80B: from the very first layer, before every 7th block (12 layers over 80 LM layers; table 10B).
- Derived check: ceil(80/7)=12, 40/4=10, 24/1=24 matches L in Table 4 (12, 10, 24). Choice is "a trade-off between memory constraints and downstream performance"; found at small scale.
- Ablation of frequency on 3B (Table 3 v, overall score): every layer 70.7; every 2nd 68.2; every 4th 68.8; single in the middle 59.8.
- Other ablations (Table 3 iv, overall): vanilla xattn (no dense) 66.9; grafting 63.1.

Transformer sizes (Table 4, App. A.1.4). Keys/values dim = D/H (128 for xattn-dense and LM, 96 for resampler); FFW hidden = 4D.
| | Resampler L/D/H | GATED XATTN-DENSE L/D/H | Frozen LM L/D/H |
| 3B | 6 / 1536 / 16 | 24 / 2048 / 16 | 24 / 2048 / 16 |
| 9B | 6 / 1536 / 16 | 10 / 4096 / 32 | 40 / 4096 / 32 |
| 80B | 6 / 1536 / 16 | 12 / 8192 / 64 | 80 / 8192 / 64 |

## 2. Perceiver Resampler (Sec. 2.1, App. A.1.1, Fig. 5 pseudocode)

Verbatim pseudocode (Fig. 5):
```
def perceiver_resampler(
    x_f,              # The [T, S, d] visual features (T=time, S=space)
    time_embeddings,  # The [T, 1, d] time pos embeddings.
    x,                # R learned latents of shape [R, d]
    num_layers,       # Number of layers
):
    """The Perceiver Resampler model."""
    # Add the time position embeddings and flatten.
    x_f = x_f + time_embeddings
    x_f = flatten(x_f)   # [T, S, d] -> [T * S, d]
    # Apply the Perceiver Resampler layers.
    for i in range(num_layers):
        # Attention.
        x = x + attention_i(q=x, kv=concat([x_f, x]))
        # Feed forward.
        x = x + ffw_i(x)
    return x
```
- Number of latents R = 64 (Sec. 2.1: "produces a fixed number of visual outputs (64)"). Output token count = number of learned latents, independent of resolution and number of frames. (Fig. 5 draws 5 for illustration.)
- Layers: 6, D=1536, 16 heads, squared-ReLU FFW; about 194M params (Table 5), identical for all three model sizes. Bigger resamplers made training unstable when scaled with the LM (App. B.3.1).
- Keys/values are the CONCAT of the flattened visual features and the latents themselves (unlike DETR/Perceiver); "slightly better".
- Time: frames sampled at 1 FPS, each encoded independently by the vision encoder; a LEARNED temporal position embedding [T,1,d] (one vector per frame, broadcast over the S spatial positions) is added; ONLY temporal, no explicit spatial grid embeddings ("did not observe improvements"; reason offered: CNN implicitly encodes spatial info). An image = single-frame video (T=1).
- Video training: 8 frames at 1 fps. Inference: 30 frames at 3 FPS, by linearly interpolating the learned temporal embeddings (App. B.1.2).
- Vision encoder: frozen NFNet-F6 (435M), contrastively pretrained (CLIP-style two-term loss) on ALIGN+LTIP; final-stage 2D grid flattened; images resized to 320x320 (contrastive pretrain at 288x288).
- Ablation (Table 3 vi, overall): Perceiver 70.7; MLP 66.6; plain Transformer 66.7.

## 3. Per-image masking (Sec. 2.3, App. A.1.3, Fig. 7)
- Text token attends (cross-attention) only to the visual tokens of the LAST PRECEDING image in the interleaved sequence, not to all previous images. Dependency on earlier images persists only through self-attention in the LM (earlier text tokens carry it).
- Formal: phi: [1,L] -> [0,N] gives for each text position the index of the last image/video before it (0 if none). x_{<=l} = {x_i | i <= phi(l)} in Eq. 1 -- NOTE the set notation reads "all images up to phi(l)" but the mask realises attention to image phi(l) only.
- Eq. 1 (main text, numbered): p(y|x) = \prod_{\ell=1}^L p(y_\ell | y_{<\ell}, x_{\leq \ell}).
- Implemented as a mask over the full text-to-image cross-attention matrix (Fig. 7: dark = visible, light = masked). Text before the first image (phi=0) sees no image.
- Generalises: trained with up to N=5 images per sequence (L=256 tokens), evaluated with up to 32 shots.
- Ablation (App. Table 10 ii): only-last vs all-previous: overall 70.7 vs 63.5 (paper text: "7.2% better"). Index tags / index embeddings for all-previous were less robust to train/test image-count changes.
- TRAINING-TIME TWIST (App. A.3.2): M3W image placement. For each webpage, with prob p_next = 1/2 the text is mapped to the NEXT image instead of the previous one (so half of training examples use the "unnatural" association). p_next 0.0 -> overall 69.6, 1.0 -> 70.4, 0.5 -> 70.7 (small win, "data augmentation" effect). Inference always uses image-before-text.
- Special tokens: <image> tag at image position, <EOC> (end of chunk) before each image and at doc end, <BOS>.

## 4. Sizes, frozen vs trainable (Table 5, App. B.1.1)
| Model | Frozen LM | Frozen vision | Trainable GATED XATTN-DENSE | Trainable Resampler | Total |
| Flamingo-3B | 1.4B | 435M | 1.2B (every) | 194M | 3.2B |
| Flamingo-9B | 7.1B | 435M | 1.6B (every 4th) | 194M | 9.3B |
| Flamingo (80B) | 70B | 435M | 10B (every 7th) | 194M | 80B |
- Frozen LM = Chinchilla 1.4B / 7B / 70B (Hoffmann et al.); frozen vision = NFNet-F6.
- Trainable per table = xattn-dense + resampler: 3B ~1.39B, 9B ~1.79B, 80B ~10.2B (derived sum; about 12.7% of 80B). Only 80B needs model sharding (16-way).
- Text of App. B.1.1 gives 1.4B / 1.8B / 10B for "additional learned parameters"; these match table xattn-dense + resampler (1.2+0.194, 1.6+0.194), not xattn-dense alone.

## 5. Training data (Sec. 2.4, App. A.3)
- M3W: ~43M webpages (datasheet: 43.3M documents, 185M images, 182 GB text); subsequence L=256 tokens, up to first N=5 images.
- ALIGN: 1.8 billion image-alt-text pairs (avg 12.4 tokens).
- LTIP (Long Text & Image Pairs): 312 million pairs (avg 20.5 tokens).
- VTP (Video & Text Pairs): 27 million short videos, ~22 s average.
- Mixture weights lambda_m: M3W 1.0, ALIGN 0.2, LTIP 0.2, VTP 0.03 (tuned at small scale, kept fixed). Gradients accumulated over all datasets (round-robin worse: 62.9 vs 70.7).
- Optim: AdamW, clip 1, wd 0 on resampler / 0.1 elsewhere, LR 0 -> 1e-4 linear over 5000 steps then constant, 500k steps. 80B: 1536 TPUv4 chips, 15 days.

## 6. Objective (Sec. 2.4, Eq. 2; numbered (2) in PDF)
\sum_{m=1}^{M} \lambda_m \cdot \mathbb{E}_{(x, y)\sim \mathcal{D}_m} \left[ -\sum_{\ell=1}^L \log p(y_\ell | y_{< \ell}, x_{\leq \ell})\right]
- m indexes the M=4 datasets; D_m the m-th dataset; lambda_m its weight; inner term = per-example NLL of the text tokens given preceding text and images (Eq. 1).

## 7. Few-shot results (Table 1, Fig. 2)
- Flamingo-80B, 32 shots, no fine-tuning, vs fine-tuned SotA (shots/K-examples in brackets): OKVQA 57.8 vs 54.4 (10K); MSVDQA 52.3 vs 47.9 (27K); Flickr30K 75.4 vs 67.4 (30K); iVQA 45.3 vs 35.4 (6K); STAR 42.2 vs 36.7 (46K); NextQA 33.5 vs 25.2 (38K). That is SIX wins of 16; RareAct (16th) has no FT baseline.
- Where FT still wins: VQAv2 67.6 vs 80.2 (444K); COCO 113.8 vs 143.3 (500K); VATEX 65.1 vs 76.3; VizWiz 49.8 vs 57.2; MSRVTTQA 31.0 vs 46.8; YouCook2 86.8 vs 138.7; VisDial 55.6 vs 75.2; TextVQA 37.9 vs 54.7; HatefulMemes 70.0 vs 79.1.
- Scaling (OKVQA / VQAv2 / COCO at 0, 4, 32 shot): 3B 41.2/49.2/73.0, 43.3/53.2/85.0, 45.9/57.1/99.0; 9B 44.7/51.8/79.4, 49.3/56.3/93.1, 51.0/60.4/106.3; 80B 50.6/56.3/84.3, 57.4/63.1/103.2, 57.8/67.6/113.8.
- "Outperforms by a large margin all previous zero-shot or few-shot methods on the 16 benchmarks", with as few as 4 examples; "the largest model better exploits larger numbers of shots".
- Fine-tuned 80B (Table 2): new SotA on 5 more tasks: VQAv2 82.0 test-dev, VATEX 84.2, VizWiz 65.7, MSRVTTQA 47.4, HatefulMemes 86.6. Fine-tune = short schedule, small LR, vision backbone unfrozen, higher resolution.
- Zero-shot prompt = 2 text-only examples with images removed.
- Ablations of freezing (Table 3 viii): unfreeze LM pretrained 62.7, random init 57.8, vs frozen 70.7 (also slower step 2.42s vs 1.74s).
- Data ablation (Table 3 i): w/o M3W 53.4; w/o image-text pairs 60.9; w/o video-text 67.3; LTIP->LAION 66.4.

## Outline assertions I could not confirm or that are wrong
- "few-shot headline: 16/32-shot beats fine-tuned SOTA on 6 of 16": CONFIRMED with 32 shots (Fig. 2 caption and Sec. 3.1 say six). But the Table 1 caption (both LaTeX and PDF) says "seven tasks"; the table data supports six. Use six; paper is internally inconsistent. Not "16/32-shot": it is 32-shot.
- "alpha init 0": true, but there are TWO alphas per block (alpha_xattn and alpha_dense), both init 0. The outline implies one gate.
- "image-attends-only-to-its-following-text": reversed subject. Text tokens attend to the last PRECEDING image; equivalently each image is visible to the text between it and the next image. Also in training, with prob 1/2 the mapping is to the NEXT image (p_next=1/2), so do not draw the training rule as strictly "preceding".
- "trainable parameter count for 80B": no single figure in the paper. Table 5: 10B xattn-dense + 194M resampler (~10.2B trainable); text says "10B additional learned parameters". Total 80B.
- "Perceiver Resampler squeezes any number of patches/frames into 64 tokens": correct (64 latents, 6 layers), and 64 is in main text only; the pseudocode leaves R symbolic.
- "Eq./Fig. 4 pseudocode": Fig. 4 is the xattn-dense pseudocode (unnumbered equation); resampler pseudocode is Fig. 5 (App. A.1.1). Numbered equations: Eq. 1 (likelihood), Eq. 2 (weighted NLL).
- Text-vs-table mismatch: App. B.1.1 text says 1.4B / 1.8B added params for 3B / 9B; Table 5 says 1.2B / 1.6B (text figures appear to include the 194M resampler).
- Flamingo-9B is "Flamingo-9B" (7.1B LM, 9.3B total); 3B = 1.4B LM.
