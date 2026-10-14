# VLMs: the papers that built them -- outline

Reader: explains everything from scratch, but knows transformers and CNNs (no attention/conv basics).
Page only (no narration). Story device: a chain of fixes -- every chapter opens with the wall the
previous generation hit, and the family tree draws "fixed by" arcs.

Page-wide color key (one meaning each):
- blue   = vision (pixels, patches, image embeddings, the vision encoder)
- yellow = language (text tokens, caption embeddings, the LLM)
- green  = the bridge / a correct match (connector, positive pairs, aligned)
- red    = loss / error / cost
- brown  = frozen or background (frozen weights, negatives, unmatched)

Anatomy used throughout: vision encoder (blue) -> connector (green) -> language model (yellow).

## Act I -- A shared space for pictures and words

### 1. ch-shared-space -- "Classifiers only know their labels" (CLIP, ALIGN delta)
Intuition: instead of predicting one of 1000 fixed labels, learn to tell which caption goes with
which image; then any sentence is a classifier.
Visual: viz-contrastive (COMPUTED toy). 5 images (blue) and 5 captions (yellow) as unit vectors in
2D; live 5x5 cosine-similarity matrix; real gradient descent on the symmetric InfoNCE loss;
temperature slider; "zero-shot" step drops a new image and classifies it by nearest prompt
"a photo of a {label}". States: before training / train (animate) / zero-shot.
Math: CLIP symmetric cross-entropy over logits = s_ij / tau (CLIP Fig. 3 pseudocode; no numbered eq).
ALIGN delta: same dual-encoder loss on 1.8B uncurated alt-text pairs; scale beats noise.
Need: CLIP dataset size (400M), temperature init + clamp (0.07, max logit scale 100), batch size
(32,768), zero-shot ImageNet ViT-L/14@336 (76.2%) vs ResNet-50 supervised, prompt template gain
(prompt engineering + ensembling), ALIGN pair count (1.8B), ALIGN filtering rules, ALIGN zero-shot
ImageNet (76.4%), ALIGN temperature learned.

### 2. ch-sigmoid -- "The softmax needs the whole batch" (SigLIP)
Intuition: softmax couples each row to every other pair (needs global normalisation, all-gather);
a sigmoid judges each pair on its own, so batches can grow cheaply and small batches work better.
Visual: viz-siglip (COMPUTED, same toy as ch1). Toggle softmax vs sigmoid; the matrix shows which
cells each loss term touches; batch-size slider shows the NxN matrix the softmax must materialise
across devices vs the sigmoid's chunked version (memory/communication are computed from N).
Math: SigLIP Eq. (1) (softmax) vs Eq. (2)? sigmoid loss with learnable t and bias b (init t'=log 10,
b=-10). SigLIP has NO LaTeX source: transcribe from PDF, loc "PDF, Sec. N, Eq. N".
Need: loss equation(s), init values, 32k sufficiency, 1M batch experiment, SigLiT 84.5% 4 TPUv4 2
days, small-batch advantage numbers (Fig./Table), chunked implementation description.

### 3. ch-patches -- "How an encoder sees" (ViT)
Intuition: cut the image into P x P patches, each patch becomes a token; token count is
(H/P)(W/P), attention cost grows with its square -- the budget every later chapter fights.
Visual: viz-tokens (COMPUTED). Sample image with patch grid overlay; resolution + patch-size
sliders; readout: tokens, attention pairs. Reused (same component) in ch-resolution.
Math: ViT Eq. 1 (patch embedding + [class] + position embeddings).
Need: ViT Eq. 1, patch sizes/variants (B/16, L/16, H/14; param counts Table 1), JFT-300M data need
(ViT underperforms ResNets on ImageNet-only pretraining), CLIP ViT-L/14 used 224 and 336.

## Act II -- Teaching a language model to see

### 4. ch-bootstrap -- "Web captions are noisy" (BLIP)
Intuition: use the model to clean its own data -- a captioner writes new captions, a filter
throws out captions that do not match their image.
Visual: viz-capfilt (TOY). Four web images with alt-text; steps: raw web text -> captioner adds a
synthetic caption -> filter (ITM score) keeps/drops each -> cleaned set. Plus a small panel of the
three objectives (ITC / ITM / LM) sharing one model.
Need: MED architecture (unimodal encoders, image-grounded encoder, image-grounded decoder, shared
params), objectives ITC/ITM/LM, CapFilt procedure, nucleus vs beam sampling finding, dataset
sizes (14M, 129M), headline gains.

### 5. ch-flamingo -- "Don't break the language model" (Flamingo)
Intuition: keep the LM frozen, insert new cross-attention layers whose output is multiplied by
tanh(alpha) with alpha = 0 at start, so on day one the model is exactly the old LM and vision is
dialled in gradually; a Perceiver Resampler squeezes any number of patches/frames into 64 tokens.
Visual: viz-gate (COMPUTED). alpha slider: tanh(alpha) curve, a text-token vector = LM path +
gated vision path, bar showing how far output moves from the frozen LM. Second panel: resampler,
slider for number of input frames/patches, output stays 64 latents.
Math: gated xattn-dense residual (Flamingo Eq./Fig. 4 pseudocode), per-image masking rule.
Need: model sizes (3B/9B/80B, frozen Chinchilla 70B), latents = 64, alpha init 0, M3W (43M pages),
ALIGN/LTIP data sizes, few-shot headline (e.g. 16/32-shot beats fine-tuned SOTA on 6 of 16),
trainable parameter count for 80B, image-attends-only-to-its-following-text masking.

### 6. ch-qformer -- "A small bridge between two frozen giants" (BLIP-2)
Intuition: 32 learned query vectors ask the frozen image encoder what matters; their outputs
become a 32-token soft prompt for a frozen LLM. Train only the bridge.
Visual: viz-qformer (COMPUTED from the paper's spec). Mask selector ITC / ITM / ITG shows the
self-attention mask over [32 queries | text tokens] (uni-modal, bidirectional, multimodal causal),
plus trainable-vs-frozen parameter bars.
Need: queries 32 x 768, Q-Former init (BERT-base), 188M params, stage 1 objectives + masks, stage 2
(FC projection into LLM, decoder vs encoder-decoder LLMs), frozen encoders (ViT-g/14 EVA-CLIP),
Flamingo80B comparison (+8.7% zero-shot VQAv2, 54x fewer trainable params), 129M images, compute.

## Act III -- From describing to conversing

### 7. ch-llava -- "Chat needs conversations to learn from" (LLaVA, LLaVA-1.5 delta)
Intuition: the simplest bridge (a linear layer) is enough if you have the right data; text-only
GPT-4 turns captions + boxes into conversations; LLaVA-1.5 adds an MLP, 336 px and short-answer
formatting prompts.
Visual: viz-stages (COMPUTED token counts + sourced structure). Stage stepper: stage 1 (align:
projector trainable, encoder + LLM frozen) -> stage 2 (instruction tuning: projector + LLM
trainable). Connector comparison: Flamingo resampler 64 / Q-Former 32 / LLaVA linear 256 /
LLaVA-1.5 MLP 576 tokens per image (computed from grid sizes), what is trained.
Math: LLaVA Eq. 1 (H_v = W . Z_v), training objective (Eq. 3, autoregressive over answer tokens).
Need: 158K split (58K/23K/77K), 595K CC3M filtered, which CLIP layer (penultimate grid), stage
epochs/steps, LLaVA-Bench relative score 85.1%, ScienceQA 92.53%, LLaVA-1.5: MLP 2-layer GELU,
336px, response format prompt text, 665K instruction mix, 1.2M total, ~1 day 8xA100, 11/12 SOTA.

## Act IV -- Seeing sharply and pointing

### 8. ch-resolution -- "336 pixels cannot read a page" (Qwen-VL -> Qwen2-VL)
Intuition: small fixed squares blur text; feed the image at its own size and merge neighbouring
patches to keep the token count in check; give each visual token a 2D (and time) position.
Visual: viz-resolution (COMPUTED, reuses viz-tokens code). Presets document page / wide
screenshot / small photo; strategy toggle fixed resize / tiling / native dynamic; readout tokens
and effective resolution. Second: viz-mrope -- position ids for [text, image grid, text] under
1D RoPE vs M-RoPE (temporal, height, width components).
Need: Qwen-VL 448px, cross-attn adapter to 256 tokens, 3-stage pipeline; Qwen2-VL 14px patches,
2x2 merge -> 1 token per 28x28 px, min/max pixels, 2D-RoPE in ViT, M-RoPE id rule, video 2 fps /
3D conv over 2 frames, sizes 2B/8B/72B, DocVQA numbers.

### 9. ch-grounding -- "Say where, not just what" (Qwen-VL boxes, Florence-2, Molmo points)
Intuition: a box is just text: quantise coordinates into bins and write them as tokens; then
detection, grounding and OCR-with-regions become next-token prediction.
Visual: viz-boxes (COMPUTED). Drag a box on a toy scene; live serialisation in three formats
(Qwen-VL "<box>(x1,y1),(x2,y2)</box>" on 0-1000; Florence-2 <loc_k> tokens; Molmo point);
bins slider shows quantisation error in pixels.
Need: Qwen-VL box format + normalisation range, ref/box special tokens; Florence-2 1000 location
bins, token forms for box/quad/polygon, DaViT encoder + enc-dec, 0.23B/0.77B sizes, FLD-5B
(5.4B annotations, 126M images), data engine loop; Molmo point format (coords 0-100?) + counting.

## Act V -- What actually matters

### 10. ch-recipes -- "Most design choices were never tested" (Idefics2, Molmo)
Intuition: controlled ablations say the backbones, resolution and data matter; the connector
style matters less; human-spoken captions beat distilling a closed VLM.
Visual: viz-ablations (SOURCED). Tabs over Idefics2 ablation tables: swap LM vs swap encoder;
cross-attn vs fully autoregressive (with/without unfreezing); pooling to 64 tokens; image
splitting; and a Molmo tab: caption source / pointing counting.
Need: Idefics2 Tables for backbone swap, architecture, pooling, splitting, sizes (8B, SigLIP-SO400M
+ Mistral-7B, 64 tokens, 980px); Molmo PixMo-Cap (spoken 60-90s, 712k images?), PixMo-Points,
Elo/human eval, 72B beats Claude 3.5 Sonnet/Gemini 1.5 Pro (which metric), no VLM distillation.

## Act VI -- Other paths, and where it all lands

### 11. ch-early-fusion -- "Why not make images words?" (Chameleon)
Intuition: quantise an image into codebook ids, put them in the same sequence as text, and one
transformer reads and writes both; stability needs extra care.
Visual: viz-fusion (TOY VQ computed). Toggle late fusion (encoder -> connector -> LLM, text out)
vs early fusion (image -> codebook ids interleaved with text, image or text out); a toy vector
quantiser snaps patch colours to the nearest of K codebook entries.
Need: tokenizer 512x512 -> 1024 tokens, codebook 8192, BPE vocab 65,536, sizes 7B/34B, tokens
trained (~10T?), QK-Norm + z-loss + norm reordering, output-norm divergence, human eval.

### 12. ch-qwen3 -- "Putting it together" (Qwen3-VL)
Intuition: a 2025 VLM is the chain of fixes assembled: SigLIP-style encoder at native resolution,
MLP merger, M-RoPE (now interleaved), DeepStack multi-layer features, text timestamps for video,
256K context.
Visual: viz-anatomy (SOURCED structure). Clickable anatomy; each part names the chapter/paper that
introduced it; DeepStack toggle shows ViT features from several layers injected into early LLM
layers.
Need: interleaved-MRoPE, DeepStack (which ViT layers -> which LLM layers), text-based timestamps,
256K native, dense 2B/4B/8B/32B + MoE 30B-A3B / 235B-A22B, vision encoder (SigLIP-2 init?),
ablation numbers for DeepStack.

## Closing sections
- ch-timeline: family tree by date, lanes (contrastive / bridging / instruction / resolution+
  grounding / recipes / early fusion), "fixed by" arcs.
- ch-summary: one paragraph.
- ch-deeper: per paper, the section/equation to read first; plus unsourced pointers to topics
  this page skips (video, GUI agents, hallucination benchmarks).

## Corrections from extraction (2026-09-29), supersede the chapter specs above
- ch1: CLIP has no numeric supervised ResNet-50 figure; say "matches the original ResNet-50". ALIGN loss
  sums the two directions, label smoothing 0.1, temperature init 1.0 -> ~1/64; ALIGN data is lightly
  frequency-filtered, not "uncurated". Toy: cold training from random starts jams ~half the time in 2D;
  seed 65 used, stated in caption.
- ch2: SigLIP equations are unnumbered (cite Sec. 3.1/3.2/3.3 + Algorithm 1). Peak batch 32k (sigmoid)
  vs 98k (softmax); 1M experiment is SigLiT only. Toy shows b=-10 init is what makes the sigmoid train
  (10/10 seeds vs 3/5 mean pairs at b=0): use as the prediction prompt.
- ch3: ViT has no L/14; token counts are computed (N = HW/P^2). ViT-L < ViT-B on ImageNet-only
  pretraining; similar on 21k; big models win only on JFT-300M.
- ch5: TWO gates per block (alpha_xattn, alpha_dense), both init 0. Text attends to the LAST
  PRECEDING image (p_next=0.5 random next-image mapping in M3W training). 32-shot beats fine-tuned SotA
  on 6 of 16 (not 7). xattn every 1/4/7 blocks for 3B/9B/80B.
- ch6: +8.7 is absolute points (65.0 vs 56.3). 54x = 10.2B / 188M; Table 2 trainable 103-108M: show
  188M with the paper's own framing and note the discrepancy or avoid the ratio.
- ch7: token counts 256/576 computed, not stated. MLP activation not in paper. "SoTA on 11 of 12
  evaluated". 92.53 is LLaVA+GPT-4 judge (LLaVA alone 90.92). Pretrain set 595K (LLaVA) vs 558K (1.5).
- ch8: title fix: 336 px is LLaVA-1.5; Qwen-VL is 224 (stage 1) / 448. Qwen-VL adapter: one xattn
  layer, 256 queries. Qwen2-VL: 224x224 -> 66 tokens (64 + 2 special); M-RoPE next id = max over
  (t,h,w) + 1; LLM sizes 1.5B/7.6B/72B + 675M ViT; min/max pixels only in the ablation.
- ch9: Qwen-VL boxes are plain digits on [0,1000) in <box>(x1,y1),(x2,y2)</box> with <ref>; Florence-2
  1,000 bins, token spelling not in LaTeX (figure shows "<loc_586>"); Molmo points are plain text
  <point x="10.0" y="10.0" alt="...">, scale 0-100; point-then-count helps counting.
- ch10: connector DOES matter in Idefics2's appendix (linear 44.5 / MapNet 51.8 / perceiver 60.3, check
  it compiles); what matters less: xattn vs fully-AR once unfrozen, tokens >64. Molmo human captions
  MATCH GPT-4o captions (Elo 1014 vs 1018), beat ShareGPT4V. Molmo-72B beats Claude 3.5 Sonnet / Gemini
  1.5 Pro on its own Elo + 11-benchmark average, not on Chatbot Arena.
- ch11: Chameleon 9.2T tokens seen (2.1 epochs); overall human-eval win rate vs Gemini-Pro+ 58.8%
  (60.4% is the mixed-modal row), vs GPT-4V+ 51.6%; norm reordering only for 34B, dropout for 7B.
- ch12: DeepStack = 3 ViT levels added to first 3 LLM layers (ViT indices unstated); interleaved MRoPE
  has no equation/ablation; SigLIP-2 encoder; timestamps as text "<3.0 seconds>"; 4 dense + 2 MoE.

## Added at checkpoint 2 (user request): chapter 8, ch-training-step
Final chapter order (ids): 1 ch-shared-space, 2 ch-sigmoid, 3 ch-patches, 4 ch-bootstrap, 5 ch-flamingo,
6 ch-qformer, 7 ch-llava, 8 ch-training-step, 9 ch-resolution, 10 ch-grounding, 11 ch-recipes,
12 ch-early-fusion, 13 ch-qwen3; then ch-timeline, ch-summary, ch-deeper.

### 8. ch-training-step -- "One training step, end to end" (LLaVA, Flamingo, Molmo, Qwen2-VL, CLIP)
User asked: how training with images and text works together; how a batch is formed; a pipeline.
Intuition: a VLM training step is ordinary next-token prediction on one long sequence in which the
image has been turned into tokens; the art is in how examples are sampled, spliced, batched and masked,
and which weights receive gradients.
Visual: viz-pipeline (COMPUTED from the papers' specs; toy examples). Mode toggle:
 A. contrastive step (CLIP): N pairs -> two encoders in parallel -> N image + N text vectors -> N x N table
    -> loss; every other caption in the batch is a negative, so batch size is part of the method.
 B. generative step (LLaVA-style), stage stepper, one cue button per stage:
    1 sample: draw examples from a dataset mixture (Flamingo Eq. 2 weights 1.0/0.2/0.2/0.03; gradients
      accumulated over all datasets beat round-robin 70.7 vs 62.9)
    2 image -> resize -> patch grid -> encoder -> connector -> visual tokens (576 for 336 px / 14 px patch,
      computed)
    3 text -> conversation template (LLaVA Table 2: Human: ... <STOP> Assistant: ... <STOP>) -> tokens
    4 splice: the <image> placeholder is replaced by the visual tokens -> one sequence per example
    5 batch: three examples of different lengths; toggle pad-to-longest (grey pad tokens, waste % computed)
      vs pack (concatenate + block-diagonal attention mask; Molmo packs all annotations of one image with
      masks so they cannot see each other: images processed -2/3, training time more than halved;
      Qwen2-VL packs images of different resolutions)
    6 forward + loss mask: next-token targets; loss only on assistant answers and their <STOP> (LLaVA
      Eq. 3 / Sec. 4.2); visual tokens and human turns get no loss (show the label row with ignored cells)
    7 backward: gradients reach only trainable parts; stage toggle (LLaVA stage 1: connector only;
      stage 2: connector + LLM; encoder frozen) -- frozen parts brown.
Math: LLaVA Eq. 3 (autoregressive likelihood over answer tokens), Flamingo Eq. 2 (weighted mixture).
Padding/packing mechanics beyond what Molmo/Qwen2-VL state are "common practice" and labelled so.

## Revision 2026-09-30 (user request): architecture + gradient flow, CLIP first
ch-shared-space gains two sections between "Temperature" and "Zero-shot":
- "Inside CLIP: one forward pass, shape by shape": viz-clipflow (COMPUTED). Two lanes (image blue, text yellow) ->
  logits -> loss; stepper 1 Batch .. 7 Loss, 8 Gradients, 9 Update; model pills ViT-B/32, B/16, L/14, L/14@336px
  change the computed shapes (CLIP appendix "CLIP-ViT hyperparameters"). Backward stage marks every trainable box red:
  nothing is frozen (the contrast for every later chapter's brown boxes).
- "Backward: where the gradient goes": viz-clipgrad (TOY, exact gradient). One draggable photo; green pull toward its
  caption, brown pushes from captions weighted by (p+q)/2; "Apply the normalization" splits radial vs tangent.
  Math: eq-clip-grad (derived on the page; CLIP prints no gradient).
Template for the other model chapters if the user approves: forward with shapes, then which weights get gradient.

## Revision 2026-09-30 (b): every architecture chapter gets "Inside <model>: forward, then backward"
(forward stepper with computed shapes, backward with frozen/trainable per training stage), toys where the gradient's
shape is the insight (SigLIP per-pair, Flamingo zero-gate), claim cards for ch-recipes (analysis papers).
