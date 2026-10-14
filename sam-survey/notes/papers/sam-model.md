# SAM (Segment Anything) -- model, task, training (architecture lens)

Scope: ch-promptable, ch-sam-arch, interlude ch-training. Source read: `sources/sam/latex/segany.tex`
(arXiv 2304.02643v1), PDF for figure/table numbers, official README.

## 1. Identity
- Title: Segment Anything. arXiv 2304.02643v1, submitted 2023-04-05 (abs.html: "Submitted on 5 Apr 2023"; only v1 on the page).
- Authors: Alexander Kirillov (project lead, joint first) ... Ross Girshick; 12 authors, Meta AI Research, FAIR.
  Joint first authors: Kirillov, Mintun, Ravi, Mao.
- Venue: none stated in v1 (the tarball ships `iccv.sty`, but the text names no venue).
- Repo: github.com/facebookresearch/segment-anything (Apache 2.0). Checkpoints `vit_h` (default), `vit_l`, `vit_b`.

## 2. Core problem
Build a foundation model for segmentation that solves new segmentation tasks zero-shot through prompts.
This needs a task that is both a pre-training objective and a general interface (Sec. 2).

## 3. What prior methods did and why that hurt (per the paper)
- Multi-task segmentation systems run a fixed task set; train and test tasks are the same (Sec. 2 "Related tasks").
- Interactive segmentation aims for a valid mask "after enough user input"; SAM must give a valid mask for any
  prompt, even an ambiguous one (Sec. 2 "Pre-training").
- A single-output model "will average multiple valid masks if given an ambiguous prompt" (Sec. 3, App. A).
- Prior interactive methods do "only one or a few interactive steps per optimizer update" (App. A, Training algorithm).

## 4. The one key idea
Split the model: a heavy image encoder runs once per image; a light prompt encoder + mask decoder runs per
prompt (~50ms in a browser on CPU). The decoder outputs 3 masks plus a predicted IoU for each, and training
back-propagates only the lowest-loss mask, so an ambiguous prompt gets several valid answers, not an average.

## 5. Mechanism, step by step
1. Resize the image (our reading: long side to 1024; App. A says only "rescaling the image and padding the shorter side") and pad the shorter side -> 1024x1024x3 (App. A).
2. ViT-H/16 (MAE init), 14x14 windowed attention, 4 equally spaced global-attention blocks (ViTDet style)
   -> 16x downscale -> 64x64 grid of tokens (App. A).
3. Neck: 1x1 conv to 256 ch, LN, 3x3 conv 256 ch, LN -> image embedding 256x64x64 (App. A).
4. Sparse prompts -> 256-d tokens. Point = Fourier positional encoding (Tancik et al.) of its location + one of
   two learned embeddings (foreground / background). Box = 2 tokens: PE(top-left) + learned "top-left" embedding,
   PE(bottom-right) + learned "bottom-right" embedding. Text = CLIP text encoder (App. A, App. D.5).
5. Dense prompt (mask) at 4x lower res than input (256x256, our reading) -> conv 2x2 s2 to 4 ch -> conv 2x2 s2 to
   16 ch -> conv 1x1 to 256 ch, GELU + LN between layers -> 64x64x256, added element-wise to the image embedding.
   No mask -> a learned "no mask" embedding is added at every location (App. A).
6. Insert learned output tokens into the prompt-token set: 3 mask tokens (multi-mask), a 4th mask token
   (single-mask output when more than one prompt), and an IoU token (App. A). Count 3+1+1 = 5 is our sum.
7. Two-way transformer, 2 layers. Each layer, 4 steps: (1) self-attention on tokens; (2) cross-attention tokens
   (queries) -> image embedding; (3) point-wise MLP on each token; (4) cross-attention image embedding (queries)
   -> tokens. Image embedding = 64^2 = 4096 vectors of 256-d. Residual + LN + dropout 0.1 on each sub-block.
   Image PE is added to the image embedding at every attention; the full original prompt tokens (with PE) are
   re-added to token queries and keys at every attention (App. A, Fig. 14 caption).
8. Upsample the image embedding 4x with two transposed convs (2x2, s2; 64 then 32 ch; GELU; LN between)
   -> 32x256x256 (4x downscaled from input) (App. A).
9. Tokens attend once more to the image embedding (App. A). Our reading: this final token->image attention
   uses the 64x64 decoder-output embedding, not the upscaled one; the paper does not say which (TODO code).
10. Each mask token -> 3-layer MLP -> 32-d vector; mask logits = point-wise dot product with the 32x256x256
    upscaled embedding -> 256x256 per mask (App. A).
11. IoU head: small head on the IoU token predicts one IoU per mask (App. A).
12. Output rule: a single prompt -> return the 3 multi-mask outputs (4th never returned). More than one prompt ->
    return only the 4th mask (App. A, "Making the model ambiguity-aware").
13. Ranking: masks are ranked by predicted IoU; the eval default uses the most confident mask (Sec. 7.1).

Gradient: everything is trained, image encoder included (initialized from MAE ViT-H; layer-wise LR decay 0.8,
App. A recipe). No frozen part is stated. Signal: focal + dice on masks, MSE on IoU predictions.

## 6. Equations
The LaTeX source has NO equation environments. The loss is prose only (App. A, "Losses" and
"Making the model ambiguity-aware"). Paper wording:
- "linear combination of focal loss and dice loss in a 20:1 ratio of focal loss to dice loss"
- "compute the loss between the ground truth and each of the predicted masks, but only backpropagate from the lowest loss"
- "IoU prediction head is trained with mean-square-error loss between the IoU prediction and the predicted mask's
  IoU with the ground truth mask. It is added to the mask loss with a constant scaling factor of 1.0."
- "auxiliary deep supervision after each decoder layer is unhelpful" (so no per-layer losses).

Transcription (our reading, not in the source):
```
L_mask(k) = 20 * Focal(M_k, G) + 1 * Dice(M_k, G)
k* = argmin_k L_mask(k)                         (k over the 3 masks; single-prompt rounds only)
L = L_mask(k*) + 1.0 * MSE(s_hat, IoU(M, G))
```
- M_k: predicted mask k (logits -> probabilities); G: ground-truth mask; s_hat: predicted IoU score(s).
- Our reading, not stated: whether the MSE covers all 3 masks or only k*; how losses over the 11 rounds are
  combined; whether gradients flow through the fed-back logits; focal alpha/gamma.
- Multi-prompt rounds output one mask (4th token), so no min there (our reading from the output rule).

Custom macros used in quoted source: `\x` = `\times`, `\app` = `~`, `\expnum{8}{-4}` = 8e-4,
`\lr`/`\ld`/`\wtd`/`\drp` = lr / ld / wd / dp.

## 7. Config table (shape-computable)
Status: P = stated in paper; R = our reading / arithmetic from P; C = code-only (not downloaded, TODO);
V = standard ViT-H config from the ViT paper, not restated in SAM.

| Stage | Item | Value | Status | Loc |
|---|---|---|---|---|
| input | resolution | 1024x1024x3, rescale + pad shorter side | P | App. A |
| encoder | backbone | ViT-H/16, MAE pre-trained | P | App. A |
| encoder | patch | 16x16 -> 64x64 = 4096 tokens | P (16x downscale, 64x64) | App. A |
| encoder | width / depth / heads | 1280 / 32 / 16 | V, uncertain | ViT paper |
| encoder | attention | 14x14 windowed + 4 equally spaced global blocks | P | App. A |
| encoder | global block indices | e.g. 7,15,23,31 if depth 32 | C | -- |
| encoder | window padding | 64 -> 70 (64 not divisible by 14) | C/R | -- |
| encoder | relative pos. embeddings | | C | -- |
| neck | 1x1 conv -> 256, LN, 3x3 conv 256, LN | 64x64x256 | P | App. A |
| prompt | token dim | 256 | P | App. A |
| prompt | point | Fourier PE + learned fg/bg embedding (2 learned) | P | App. A |
| prompt | box | 2 tokens: PE + learned top-left / bottom-right | P | App. A |
| prompt | PE type | random Fourier features (Tancik et al.) | P (cite) | Sec. 3, App. A |
| prompt | PE frequency count, scale | | C | -- |
| prompt | padding point when no box | | C | -- |
| prompt | text | CLIP ViT-L/14@336px text encoder (exploratory) | P | App. D.5 |
| mask prompt | input | 4x lower than input -> 256x256x1 | P (4x) / R (256) | App. A |
| mask prompt | conv1 | 2x2 s2, 4 ch -> 128x128x4 | P / R shape | App. A |
| mask prompt | conv2 | 2x2 s2, 16 ch -> 64x64x16 | P / R shape | App. A |
| mask prompt | conv3 | 1x1, 256 ch -> 64x64x256; GELU + LN between | P | App. A |
| mask prompt | none | learned "no mask" embedding added everywhere | P | App. A |
| decoder | output tokens | 3 mask + 1 single-mask + 1 IoU = 5 | P / R count | App. A |
| decoder | tokens for 1 point | 1 prompt + 5 output = 6 | R (C may add a pad token) | -- |
| decoder | tokens for 1 box | 2 prompt + 5 output = 7 | R | -- |
| decoder | layers | 2 | P | Sec. 3, App. A |
| decoder | embed dim | 256 | P | App. A |
| decoder | heads | 8 (all attention layers) | P | App. A |
| decoder | cross-attn q/k/v dim | 128 (2x reduction) | P | App. A |
| decoder | MLP hidden | 2048 (tokens only; "rarely greater than 20" tokens) | P | App. A |
| decoder | dropout | 0.1 at training | P | App. A |
| decoder | image as set | 64^2 vectors of 256-d | P | App. A |
| decoder | self-attn q/k/v dim | | C (not stated; 256 by our reading) | -- |
| upscale | 2 transposed convs 2x2 s2 | 64x64x256 -> 128x128x64 -> 256x256x32 | P (ch) / R (shapes) | App. A |
| head | final token->image attention | once, after decoder | P | App. A |
| head | hypernetwork MLP | 3 layers, out dim 32 | P (3 layers) / R (32 = upscaled ch) | App. A |
| head | MLP hidden dim; one MLP per mask token or shared | | C | -- |
| head | mask logits | 256x256 per mask (4x downscaled) | P / R | App. A |
| head | upsample to 1024 / crop pad / resize to original | | C (paper silent) | -- |
| head | IoU head layers / hidden | "small head" | C | App. A |
| outputs | multi-mask | 3 masks + 3 IoU scores for a single prompt | P | App. A |
| outputs | single-mask | 4th mask only, for >1 prompt | P | App. A |
| loss | focal:dice | 20:1 | P | App. A |
| loss | IoU | MSE, weight 1.0 | P | App. A |
| loss | multi-mask | backprop lowest-loss mask only | P | Sec. 3, App. A |
| loss | deep supervision | none (unhelpful) | P | App. A |
| train-loop | rounds | 11 = 1 initial + 8 sampled points + 2 no-point | P | App. A |
| train-loop | first prompt | fg point or box, p = 0.5 each | P | App. A |
| train-loop | first point | uniform over GT mask | P | App. A |
| train-loop | box noise | std 10% of side length per coordinate, max 20 px | P | App. A |
| train-loop | next point | uniform over error region; FN -> fg, FP -> bg | P | App. A |
| train-loop | feedback | unthresholded logits of the highest-predicted-IoU mask | P | App. A |
| train-loop | no-point rounds | one at a random place among the 8, one always last | P | App. A |
| train-loop | max tested | up to 16 points; diminishing returns after 8 | P | App. A |
| train-loop | cost | decoder < 1% of encoder compute | P | App. A |
| recipe | optimizer | AdamW, beta1 0.9, beta2 0.999 | P | App. A |
| recipe | warmup | linear, 250 iterations | P | App. A |
| recipe | lr | 8e-4 after warmup; /10 at 60k and at 86666 | P | App. A |
| recipe | iterations | 90k (~2 SA-1B epochs) | P | App. A |
| recipe | batch | 256 images; up to 64 masks per GPU | P | App. A |
| recipe | GPUs | 256 (A100 per model card) | P | App. A, Table 9 |
| recipe | wd / dp / ld | 0.1 / 0.4 / 0.8 | P | App. A |
| recipe | augmentation | none (default) | P | App. A |
| recipe | mask filter | drop masks covering > 90% of image | P | App. A |
| recipe | data | SA-1B automatic masks only (default) | P | Sec. 7 Implementation, 7.6 |
| recipe | ViT-B / ViT-L | 180k it, batch 128, 128 GPUs; lr 8e-4 / 4e-4; ld 0.6 / 0.8; wd 0.1; dp 0.6 / 0.4 | P | App. A |
| recipe | stage-1/2-only runs | large-scale jitter, scale [0.1, 2.0] | P | App. A |
| compute | training | 256 A100 GPUs for 68 hours; ~6963 kWh; ~2.8 t CO2 | P | Table 9 (model card) |
| runtime | prompt enc + decoder | ~50ms, web browser, CPU, given image embedding | P | Sec. 1, Sec. 3 |
| runtime | image encoder time | not stated in v1 | -- | -- |
| sizes | ViT-B / ViT-L / ViT-H | 91M / 308M / 636M | P (axis labels; what is counted is not said) | Fig. 13 right |

Shape trace for the ArchFlow stepper (P/R from the table):
```
image 1024x1024x3
 -> patchify 16x16           -> 64x64 tokens x 1280 (width V)
 -> 32 blocks (V), windowed 14x14 + 4 global
 -> neck 1x1 conv + LN       -> 64x64x256
 -> neck 3x3 conv + LN       -> 64x64x256  = image embedding (+ dense prompt or "no mask")
mask prompt 256x256x1 -> 128x128x4 -> 64x64x16 -> 64x64x256
sparse: point -> 1x256 ; box -> 2x256 ; + output tokens 5x256
decoder x2: tokens Nx256 <-> image 4096x256 (cross-attn inner 128, 8 heads)
 -> final token->image attention
upscale: 64x64x256 -> 128x128x64 -> 256x256x32
mask token (x3 or x1) -> MLP -> 32-d ; dot with 256x256x32 -> 256x256 logits per mask
IoU token -> head -> 1 score per mask
256x256 -> (code-only) upsample to 1024x1024, remove padding, resize to original
```

## 8. Results worth remembering (scope of these chapters)
- Fig. 13 left (data stages): each stage increases mIoU; automatic-only is ~0.5 mIoU below all three stages, so
  default = automatic masks only. With all stages, manual + semi-automatic masks oversampled 10x (Sec. 7.6).
  Plot read (uncertain): 1 point ~49 / ~55 / ~60 / ~59; oracle ~66 / ~71 / ~73 / ~72 (manual, +semi, +auto, auto only).
- Fig. 13 middle (data volume, uniform subsample 0.1M / 1M / 11M images): large decline at 0.1M; 1M (~10%,
  ~100M masks) comparable to 11M. Lines: 2, 3, 5 points and 1 point (oracle); no non-oracle 1-point line.
  Plot read (uncertain): 5 points ~79 / ~82 / ~82; oracle ~71 / ~72 / ~72.
- Fig. 13 right (encoder): ViT-H >> ViT-B, only marginal over ViT-L; "further image encoder scaling does not
  appear fruitful". Plot read (uncertain): 1 point ~57.5 / ~58.6 / ~59.2; oracle ~69 / ~71.7 / ~72.2.
- Data-engine anecdote (Sec. 4): encoder scaled from ViT-B to ViT-H during stage 1; model retrained 6 times.
- App. B: the data-generation SAM differs: stage 1+2 data only, 177656 iterations, LSJ, point + mask prompts
  only (no boxes), 4 points per mask (vs default 9), 3 decoder layers instead of 2.

## 9. Limitations stated (Sec. 8)
Misses fine structures; hallucinates small disconnected components; boundaries less crisp than "zoom-in"
methods (FocalClick); dedicated interactive methods beat it with many points (SimpleClick); overall not
real-time with the heavy encoder; text-to-mask exploratory; unclear how to prompt semantic/panoptic segmentation.
Later fixes in this survey's sources (not verified here): HQ-SAM (fine structures), efficient encoders
(MobileSAM, EfficientSAM, EfficientViT-SAM) for the encoder cost.

## 10. Outline feedback
1. ch-sam-arch "focal + dice on the best of 3 masks only" holds only for single-prompt rounds. With >1 prompt the
   4th token gives the only mask and the loss is ordinary (App. A). In the 11-round loop, round 1 has 3 outputs +
   min; rounds 2-11 have points + fed-back mask, so 1 output (our reading). The ch-training stepper must show the
   switch from 3 outputs to 1. Our reading: a box (2 tokens) is one prompt, so a box-first round 1 has 3 masks.
2. No equations in the source. Any loss formula on the page is a transcription, labelled "our reading".
3. ch-sam-arch shapes: the outline's "ViT-H ... 1280" width is not in the SAM paper; mark as ViT-H standard.
   Final upsample 256x256 -> image size is code-only.
4. Decoder count: outline says "two-way transformer (2 layers)"; add the extra final token->image attention.
5. ch-clicks contrast: SAM samples the next point uniformly over the error region, not at the center of the
   largest error region (RITM style / the ch-clicks toy). The first training point is uniform over the mask;
   the "center" point is the eval protocol (Sec. 7.1). Worth a one-line contrast in ch-training.
6. ch-promptable: the paper's eval returns the most confident (highest predicted IoU) mask; "oracle" = best of 3
   vs GT (Sec. 7.1, 7.6). Good for the "one output only" button.
7. ch-engine owner: the automatic-stage model is a special SAM (App. B, see Sec. 8 above). Also note the 32x32
   grid, predicted-IoU filter 88.0, stability 95.0, NMS 0.7 (App. B) belong to ch-engine.
8. Runtime: the paper gives no image-encoder time; do not use the project-page number unless sourced.

## TODO (code-only values; code not downloaded, do not guess)
- ViT-H width/depth/heads (V above), global-attention block indices, window padding, rel-pos embeddings.
- Fourier PE: number of frequencies, Gaussian scale; whether PE dim is 128 sin + 128 cos.
- Pad "not-a-point" token when only points are given.
- Self-attention inner dim in the decoder (paper only states the 2x reduction for cross-attention).
- Hypernetwork MLP hidden dim; separate MLP per mask token or shared.
- IoU head depth and hidden dim.
- Focal loss alpha/gamma; dice smoothing.
- Upsample of 256x256 logits to 1024x1024, removal of padding, resize to original; threshold 0 on logits
  (App. B uses "thresholding logits at 0" for AMG).
- Which embedding (64x64 or upscaled) the final token->image attention uses.
- Image-encoder runtime and hardware.
