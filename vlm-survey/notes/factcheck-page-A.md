# Fact-check of the page, part A: ch-shared-space, ch-sigmoid, ch-patches, ch-bootstrap, ch-flamingo, ch-qformer, ch-llava

Scope and method. Read parts/ch-*.html and parts/viz-*.js for the seven chapters (confirmed verbatim in explainer.src.html).
Every data-claim span was checked against the primary source, not claims.json: LaTeX in sources/<paper>/src (CLIP, ALIGN,
ViT, BLIP, Flamingo, BLIP-2, LLaVA, LLaVA-1.5) and the SigLIP PDF (pages 1-14 read). Display equations were compared symbol
by symbol with the paper or its pseudocode (CLIP Fig. 3 pseudocode.pdf, Flamingo fig4/fig3 pseudocode PDFs, SigLIP Alg. 1).
The toys were re-run in node with build/sim/pe.cjs (its contrast/rng are byte-identical to explainer.head.part): 400 steps,
lr 0.05, seed 65, exactly as the figure code does. All 247 claim spans in these chapters (repeats counted) contain their claim value and none
uses an uncertain claim.

Severity: wrong / misleading / unlabelled / nit. Items 1-10 are the ones that matter; 11-13 are unlabelled numbers; the rest
are nits.

## Problems

### 1. wrong. ch-shared-space, parts/ch-shared-space.html (Temperature section); toy parts/viz-contrastive.js
Page: "Hot, the pairs drift together only loosely and the loss stalls high, because even a perfect answer cannot earn a
confident probability. Cold, every pair locks tight and the pairs spread evenly around the circle, as far from each other's
captions as they can get."
What the toy actually does (seed 65, the only start the buttons use): Train hot (tau = 1) ends with every pair on top of its
caption (diagonal cosines 1.00 1.00 1.00 1.00 1.00) and loss 0.87. Train cold (tau = 0.07) ends with loss 0.002 but two pairs
left open (car and pizza, diagonal cosine 0.91, about 24 degrees apart) and uneven spacing: gaps between neighbouring pairs
0.96 / 0.96 / 1.58 / 1.17 / 1.63 rad against 1.26 for even spacing (hot: 0.96 / 0.86 / 1.58 / 1.31 / 1.56). The screenshots
build/viz/viz-contrastive-dark-2-btn-contrastive-hot.png and -3-btn-contrastive-cold.png show exactly this, so a reader who
presses the buttons sees the opposite of the prose. What is true: hot keeps the loss high and the table washed out; cold drives
the loss to about zero, after which the gradient vanishes and training stops rearranging the points.
Replace with: "Hot, the pairs still end up on top of their captions, but the loss stalls high and the table stays washed out:
when cosines are divided by 1, even a perfect layout cannot earn a confident probability. Cold, the loss falls to almost zero
within a few dozen steps; once every photo's own caption wins clearly the gradient all but vanishes, so from this start two
pairs (car and pizza) are left slightly open and the spacing around the circle stays uneven."

### 2. misleading. ch-sigmoid, parts/ch-sigmoid.html (figcaption of viz-siglip); count logic in parts/viz-siglip.js stats()
Page: "From this start the bias-0 run ends with three of five pairs matched; over 200 random starts it averages about three,
against almost five for bias -10."
The count is `row.indexOf(Math.max(...row)) === i`, which gives exact ties to the first caption. In the bias-0 run from seed
65 the dog, cat and tree photos end at exactly the same angle (1.79937546) and their captions at exactly the same angle
(0.45375409), so all three rows are exact ties; the dog photo is "matched" only by tie-breaking (it has index 0). The chapter's
own prose says "those three photos can no longer tell their captions apart". Counting a match only when the own caption is
strictly closest: seed 65 gives 2 of 5; over seeds 1-200 the bias-0 mean is 1.54 (argmax count 3.02), the bias -10 mean is
4.88 (argmax 4.92). The result does not depend on the tolerance (same numbers for 1e-9 through 1e-2). The readout shows the
same inflated "3 of 5 photos sit closest to their own caption".
Replace with: "From this start the bias-0 run ends with two of five pairs matched (the three collapsed photos tie between
their three captions); over 200 random starts it averages about one and a half, against almost five for bias -10." In
viz-siglip.js stats(), count `right` only when `row.every((s, j) => j === i || row[i] - s > 1e-9)`. viz-contrastive.js readout()
uses the same argmax count and its caption admits collapses, so give it the same rule.

### 3. misleading. ch-sigmoid, parts/ch-sigmoid.html (Predict: what if the bias starts at 0?)
Page: "with a small starting temperature as well, it falls to <span data-claim="page-siglip-b0-t1">53.7</span>."
Source: SigLIP v4 Table 4, row "b = 0, t' = log 1: 53.7"; Sec. 4.9: "particularly noticeable when using a small temperature t'
initialization". The paper calls t a temperature, but the page defines t as the inverse temperature ("with t the inverse of
the temperature tau from chapter 1"; gloss "t: the inverse temperature"). t' = log 1 means t = 1, i.e. tau = 1, which chapter 1
calls hot ("Train hot (tau = 1)"). So in the page's own terms this is a hot start, the opposite of what "small temperature"
tells the reader.
Replace with: "with the scale t also started small (t' = log 1, so t = 1: a hot start in chapter 1's terms), it falls to
<span data-claim="page-siglip-b0-t1">53.7</span>." Related wording nits in the same chapter: "with the temperature starting at
<span data-claim="siglip-init-t">log 10 (t = 10)</span> in its log form" and the caption's "the temperature is held at SigLIP's
starting value t = 10": say "the scale t (the inverse temperature)".

### 4. wrong. ch-shared-space, parts/ch-shared-space.html (closing paragraph)
Page: "Every later model on this page starts from an encoder trained this way."
Source: BLIP v2 4-experiment.tex l.11: "The image transformer is initialized from ViT pre-trained on ImageNet". Chameleon has no
image encoder (the page's own ch-early-fusion: "The encoder and connector are gone. In their place is an image tokenizer").
Replace with: "Almost every later model on this page starts from an encoder trained this way; the exceptions are BLIP (Act II),
whose ViT starts from ImageNet classification weights, and Chameleon (Act VI), which has no image encoder at all."

### 5. misleading. ch-patches, parts/ch-patches.html (last paragraph of "No built-in eyes")
Page: "every vision encoder in the chapters that follow was pretrained on hundreds of millions of images, usually with the
matching game of chapter 1."
Source: as in item 4, BLIP's ViT is initialized from ImageNet (1.3M images) and BLIP's analyses use the 14M-image set; BLIP is
the very next chapter.
Replace with: "most vision encoders in the chapters that follow were pretrained on hundreds of millions of image-text pairs with
the matching game of chapter 1 (BLIP, next, is an exception: it starts from a ViT trained on ImageNet)."

### 6. wrong. ch-bootstrap, parts/ch-bootstrap.html (closing paragraph)
Page: "And one text transformer can both judge and write, depending only on its attention mask; BLIP-2 turns that trick into
its whole design."
Source: BLIP v2 3-method.tex l.68: "the text encoder and text decoder share all parameters except for the SA layers"; the
decoder has its own causal SA weights. The chapter itself says so (MED figure legend "Dashed causal SA: the decoder's own
weights"; Table 3: sharing SA too gives the worst IR@1, 59.5). Only BLIP-2 switches tasks by the mask alone (BLIP-2 Sec. 3.2:
three objectives "that share the same input format and model parameters. Each objective employs a different attention masking
strategy").
Replace with: "And one text transformer can both judge and write, with only its self-attention layers kept separate; BLIP-2 goes
one step further, sharing every weight and switching tasks by the attention mask alone."

### 7. misleading. ch-bootstrap, parts/ch-bootstrap.html (What cleaning buys)
Page: "BLIP pre-trains on <span data-claim="blip-data-14m">14M images</span> (COCO and Visual Genome with human captions;
Conceptual Captions and SBU with web text)"
Source: BLIP v2 4-experiment.tex l.20-22: "14M images in total, including two human-annotated datasets (COCO and Visual Genome),
and three web datasets (Conceptual Captions, Conceptual 12M, SBU captions)". BLIP-2 Sec. 3.4 lists CC3M and CC12M separately.
The omitted Conceptual 12M is the largest part of the 14M: BLIP appendix.tex l.72-74 counts COCO 113K, VG 100K, SBU 860K,
CC3M 3M, CC12M 10M images.
Replace with: "(COCO and Visual Genome with human captions; Conceptual Captions, Conceptual 12M and SBU with web text)".

### 8. misleading. ch-llava, parts/ch-llava.html (opening paragraph)
Page: "Flamingo and BLIP-2 can caption a photo or answer a short question about it, but ask one to follow an open request
("what is unusual here?", "what could I cook with these?") and it has never seen anyone do that."
Source: Flamingo v2 Fig. 1 caption (figures/teaser.tex l.404-406): "Out of the box, Flamingo is also capable of multi-image visual
dialogue (bottom)". LLaVA's evidence is about OpenFlamingo, not Flamingo (experiment.tex l.54): "BLIP-2 and OpenFlamingo focus on
describing the image, instead of following the user instruction".
Replace with: "Flamingo and BLIP-2 learned from captions and web pages, not from instructions. Flamingo can be prompted into short
dialogues, but in LLaVA's side-by-side tests BLIP-2 and OpenFlamingo (an open re-implementation of Flamingo) mostly describe the
scene when given an open request ("what is unusual here?", "what could I cook with these?")."

### 9. misleading. ch-llava, parts/ch-llava.html (after the LLaVA-1.5 scaling table)
Page: "No single row is dramatic; the gains come from stacking cheap changes."
Source: the table on the page (LLaVA-1.5 v2 tab:scaling_ablation, all rows verified): + VQAv2 data takes MME from 809.6 to
1197.0 (+387); + GQA data takes GQA from 51.4 to 62.0, with the footnote "Training images of GQA were observed during training".
Replace with: "Two rows are big jumps (adding VQAv2 lifts MME by <span data-computed>almost 400</span> points, and adding GQA's
own training data lifts GQA by <span data-computed>about 10</span>); the other changes are small, and the final model comes from
stacking them all."

### 10. misleading (minor). ch-shared-space, parts/ch-shared-space.html (ALIGN aside)
Page: "the ALIGN authors <span data-claim="align-loss-combo">minimize the sum of two losses</span> rather than their average, adds
label smoothing of <span data-claim="align-label-smooth">0.1</span>, and learns its temperature from an initial
<span data-claim="align-temp-init">1.0</span> down to <span data-claim="align-temp-converged">about 1/64</span>."
Source: ALIGN v2 align.tex l.612: "Compared to the baseline model that learns the temperature parameter (converged to about 1/64)";
that baseline is the ablation model, l.604: "EfficientNet-B5 image encoder, BERT-Base text encoder, embedding dimension 640", not
the full ALIGN model (EfficientNet-L2, BERT-Large). Also subject-verb agreement ("authors ... adds ... learns").
Replace with: "the ALIGN authors <span data-claim="align-loss-combo">minimize the sum of two losses</span> rather than their
average, add label smoothing of <span data-claim="align-label-smooth">0.1</span>, and learn the temperature from an initial
<span data-claim="align-temp-init">1.0</span> (in the paper's ablations, on a smaller B5 / BERT-Base model, it converged to
<span data-claim="align-temp-converged">about 1/64</span>)."

### 11. unlabelled. ch-shared-space, parts/ch-shared-space.html (Zero-shot section)
Page: "and averaging the embeddings of 80 differently worded prompts per class adds a further ..."
Source: CLIP v1 clip_paper.tex l.336: "On ImageNet, we ensemble 80 different context prompts". "80" is a sourced number outside any
claim span.
Replace with: add a claim (e.g. id clip-ensemble-n, value "80", CLIP v1, Sec. 3.1.4) and write
`<span data-claim="clip-ensemble-n">80</span> differently worded prompts`.

### 12. unlabelled. ch-flamingo, parts/ch-flamingo.html (closing paragraph)
Page: "the 80B model's new blocks alone are the 10B additional learned parameters quoted above"
Source: Flamingo v2 B_experiments.tex l.47 ("this accounts for 10B additional learned parameters"). The earlier mention is
wrapped; this repeat is bare.
Replace with: "the 80B model's new blocks alone are the <span data-claim="page-gate-added-80b">10B additional learned
parameters</span> quoted above".

### 13. unlabelled. ch-llava, parts/ch-llava.html (figcaption of viz-connectors)
Page: "The latent counts (Flamingo 64, BLIP-2 32) are stated in those papers; the patch counts 256 and 576 are computed from the
patch grids (224/14 = 16, 336/14 = 24, squared), not stated by LLaVA."
64 and 32 are sourced (Flamingo content.tex l.160 "a fixed number of visual outputs (64)"; BLIP-2 3-method.tex "we use 32
queries") but bare; 256, 576 and the grid arithmetic are computed but not marked. These, and the bare "t = 10" in the viz-siglip
caption, are missing from build/bare-numbers.txt because check-page.mjs (l.55-59) skips any text whose ancestor matches
'[data-claim],[data-toy],[data-computed],...,.readout', so everything inside a data-toy or data-computed figure, captions
included, is never listed; captions of toy and computed figures need a manual pass.
Replace with: "The latent counts (Flamingo <span data-claim="flamingo-latents">64</span>, BLIP-2
<span data-claim="page-qformer-queries">32</span>) are stated in those papers; the patch counts <span data-computed>256</span> and
<span data-computed>576</span> are computed from the patch grids (<span data-computed>224/14 = 16, 336/14 = 24</span>, squared),
not stated by LLaVA."

### 14. nit. ch-patches, parts/ch-patches.html (Read it by rows)
Page: "The paper's explanation is overfitting: with no built-in assumptions to fall back on, ViTs <span
data-claim="page-tokens-overfit">overfit more than ResNets with comparable computational cost on smaller datasets</span>, and a
bigger ViT has more room to do it."
Source: that sentence (04_experiments.tex l.174) comes from the second experiment (JFT subsets of 9M-300M, few-shot linear
probes, ViT against ResNets). For the Table 5 / Fig. 3 comparison the paper says only (l.160) "ViT-Large models underperform
compared to ViT-Base models, despite (moderate) regularization".
Replace with: "The paper points to overfitting: on ImageNet, Large trails Base "despite (moderate) regularization", and a second
experiment on subsets of JFT finds that ViTs <span data-claim="page-tokens-overfit">overfit more than ResNets with comparable
computational cost on smaller datasets</span>; a bigger ViT has more room to do it."

### 15. nit. ch-patches, parts/ch-patches.html (gloss of Eq. 1)
Page: "(the paper's Eq. 2 to 4 are the usual pre-norm attention and MLP blocks)"
Source: ViT v2 03_method.tex l.40-45: Eq. 2 is MSA(LN(.)) + residual, Eq. 3 is MLP(LN(.)) + residual, Eq. 4 is y = LN(z_L^0), the
readout from the [class] token.
Replace with: "(the paper's Eq. 2 and 3 are the usual pre-norm attention and MLP blocks, and Eq. 4 reads the image representation
off the [class] token)".

### 16. nit. ch-flamingo, parts/ch-flamingo.html (gloss of the resampler equation)
Page: "It is the only position information: the paper uses temporal encodings and no spatial grid encodings."
Source: A_method.tex l.25-27: "we only use temporal encodings and no explicit spatial grid position encodings; we did not observe
improvements from the latter. This rationale behind is likely that CNNs, such as our NFNet encoder, are known to implicitly
include spatial information channel-wise".
Replace with: "It is the only explicit position encoding: the paper found spatial grid encodings did not help, likely because the
NFNet's convolutional features already carry spatial information."

### 17. nit. ch-flamingo, parts/ch-flamingo.html (figcaption of viz-resampler)
Page: "because the encoder's grid size is not given in our notes"
"Our notes" means nothing to a reader. The paper gives the input size (320 x 320, B_experiments.tex l.63) but not the NFNet-F6
output grid.
Replace with: "because the paper does not state the encoder's output grid size".

### 18. nit. ch-bootstrap, parts/ch-bootstrap.html (ITC and ITM paragraphs) and parts/viz-capfilt-med.js (TXT.itm)
Page: "a slowly updated copy of the model whose soft scores replace the hard "only the diagonal is right" targets"; "ITC's scores
pick the wrong captions that look most right"; readout: "the negatives are the wrong captions the contrastive score found most
similar."
Source: BLIP 3-method.tex l.49-51: "soft labels are created from the momentum encoder as training targets" (ALBEF mixes soft and
one-hot targets); l.59-61: negatives with higher similarity "are more likely to be selected" (sampled, not the argmax).
Replace with: "whose soft scores soften the hard ... targets"; "ITC's scores make the wrong captions that look most right the
likeliest picks"; readout "the negatives are sampled, favouring the wrong captions the contrastive score found most similar."

### 19. nit. ch-bootstrap, parts/ch-bootstrap.html (green verdict box in viz-capfilt)
Page: "COCO image-retrieval R@1 after finetuning is <span data-claim="page-capfilt-ir-none">60.7</span> with no synthetic
captions"
Source: BLIP Table 2 row "None" (78.4 / 60.7) is the no-captioner, no-filter row (same as Table 1 row 1), while the Beam and
Nucleus rows are captioner plus filter.
Replace with: "... is <span data-claim="page-capfilt-ir-none">60.7</span> without CapFilt".

### 20. nit. Verbatim quotes printed as the page's own words (ch-sigmoid, ch-bootstrap)
Page: "And since <span ...>Usually b is constant as scaling |B| is achieved by increasing the number of accelerators</span>, the
per-device block ..." (capital U mid-sentence); "The frozen tower also makes SigLiT very cheap: <span ...>with only four TPUv4
chips, we train a SigLiT model that achieves 84.5% ImageNet zero-shot accuracy in two days</span>." (the paper's "we" reads as
the page's "we", which the page also uses for itself, e.g. "so we follow Algorithm 1"); ch-bootstrap "so BLIP mines hard ones:
<span ...>negatives pairs with higher contrastive similarity ...</span>" (the paper's typo reads as the page's).
Replace with: put each in quotation marks with attribution, keeping the spans: 'The paper adds that "<span ...>Usually b is
constant ...</span>", so the per-device block ...'; 'In the abstract's words, "<span ...>with only four TPUv4 chips, we train
...</span>".'; 'in the paper's words, "<span ...>negatives pairs with ...</span>".'

### 21. nit. ch-sigmoid, parts/ch-sigmoid.html
Page: "The toy exaggerates the effect (it is two-dimensional and learns only the bias)."
The toy learns the embeddings and the bias; only t is fixed (the caption says it correctly).
Replace with: "(it is two-dimensional and keeps t fixed)".

### 22. nit. ch-shared-space, parts/ch-shared-space.html (figcaption of viz-contrastive)
Page: "A circle is cramped: from some random starts, cold training jams two pairs on top of each other"
Simulation: over seeds 1-200 at tau = 0.07, 54.5% of runs end with at least one pair not strictly matched (28% at tau = 1).
Replace with: "from about half of random starts, cold training jams pairs on top of each other".

### 23. nit. Minor wording
- ch-shared-space: "an unseen photo lands near the trained dog photos" -> "near the trained dog photo" (there is one).
- ch-llava: "written for it by a language model that never saw a single picture" -> "that was never shown the pictures" (LLaVA
  used "language-only GPT-4", approach.tex l.56; that says nothing about how GPT-4 itself was trained).

### 24. nit. Toy parts inside "Computed" figures
viz-tokens (random, untrained E; 16 of D outputs drawn) and viz-connectors (text length 40 is a toy value; schematic stacks) carry
only the Computed badge. Both captions say what is toy, so nothing is hidden, but the rule says a toy or illustrative viz carries
the Toy badge. Consider "Computed (with toy parts)" or adding the Toy badge.

## Verified correct (brief)

- ch-shared-space: CLIP abstract quote; 400 million pairs and 500,000 queries (Sec. 2.2); 32,768 batch; temperature
  parameterisation, 0.07 init and clip at 100 (Sec. 2.3, 2.5); ViT-L/14@336px as "CLIP" (Sec. 2.5); prompt template, +1.3%, +3.5%
  (Sec. 3.1.4); 76.2 and the ResNet-50 sentence (Sec. 3.1.3, Table 1). Loss equation matches the Fig. 3 pseudocode (symmetric
  mean of two cross-entropies over cosine logits) and ALIGN Eq. 1-2. ALIGN: 1.8B, both text filters, "4x size" (Sec. 6.2), 76.4
  vs 76.2 (Table 4), sum of losses, smoothing 0.1, init 1.0. "Same month" holds (arXiv 2021-02-11 and 2021-02-26). Toy: the loss
  is CLIP's symmetric loss and training is plain gradient descent on it, as claimed; the zero-shot readout (85% dog) comes from
  the stated rule.
- ch-sigmoid: Sec. 3.1 softmax equation, Sec. 3.3 chunked equation (labels A/B/C, "swap negs across devices") and Alg. 1 all
  match; the Sec. 3.2 sign handling is correct (printed e^{z(-t x.y + b)} vs Alg. 1 logits = t x.y + b). Init t' = log 10,
  b = -10, the imbalance and prior quotes (Sec. 3.2); Table 4 (62.0 / 63.0 / 61.7 / 53.7, B, 8k, 900M); Fig. 1 mock (3 devices,
  batch 12, 4 x 4); |B|^2 to b^2, "Usually b is constant", permutes vs all-gathers (Sec. 3.3); 4096 vs 2048 (Sec. 4.2); <16 k
  (Sec. 1, 4.1); 51.5 vs 47.7 (Table 5, 3B seen); 77.2 vs 74.6 (Table 8, 3B seen); 32 k vs 98 k, softmax not better (Sec. 4.2,
  Table 5); 1 M, saturation at 32 k, peak at 256 k, precomputed ViT-g embeddings (Sec. 4.1); 84.5% in two days on 4 chips
  (abstract); the from-scratch maximum is 307 k, so "no from-scratch SigLIP at a million" holds. Toy numbers "0 of 25", "12 of
  25, 9 wrongly" and the -10 run (5 of 5, evenly spaced) reproduce.
- ch-patches: N = HW/P^2, trainable linear projection, 1D learnable positions, "carry no information", "arbitrary sequence
  lengths", 2D interpolation (Sec. 3.1-3.2); Eq. 1 symbol by symbol; model naming, 86M / 307M / 632M, D = 768 (Table 1); 224
  pretraining (App. B.1); 512 and 518 fine-tuning (Sec. 4.1); the three dataset sizes (Sec. 4.1); all eight Table 5 values and
  the 384 fine-tuning note; "few percentage points", "trumps inductive bias" (Sec. 1). CLIP ViT-L/14 224 vs 336 (Table 20).
  Computed counts: 196, 49 -> 196, 2,401 -> 38,416, 768, 1,369 and about 1.9 million, 256 vs 576.
- ch-bootstrap: MED structure (unimodal, image-grounded encoder with [Encode], decoder with [Decode]; share all but SA, and the
  quoted reason) and the MED figure's sharing links; ITC momentum encoder, ITM linear head, hard-negative sentence, LM label
  smoothing 0.1; one pass through the ViT and three through the text transformer; CapFilt definitions (Sec. 3.3); p = 0.9
  (Sec. 4.3). Table 1 (60.7 / 61.5 / 62.0 / 63.1; 129M ViT-B 62.0 -> 64.3), Table 2 (19% / 25%, 61.9 / 63.1), Table 3 (224M 59.5,
  361M 60.5, 252M 60.7, the best of four rows), Table 4 (8% vs 25%, 62.2 vs 63.1), 115M LAION, BERT-base init; table numbers
  match the compiled order. The toy CapFilt readouts match its hand-set scores (13 vs 23 content words; 6 of 10 texts kept).
- ch-flamingo: Fig. 4 and Fig. 5 pseudocode match the displayed equations line by line (resampler x renamed z, as glossed); the
  two gates, alpha init 0 and the init-match quote (Sec. 2.2); 1.4B / 7.1B / 70B, 435M, 3.2B / 9.3B / 80B, 194M, 10B for 80B,
  frequencies 24 of 24 / 10 of 40 / 12 of 80, resampler 6 / 1536 / 16 (App. tables); ablations 53.4, 66.5, 66.6, 66.7, 62.7,
  63.5 vs 70.7 (3B, 4 shots, overall-score definition, Sec. 3.3 and the appendix ablation table); forgetting quote; 8 frames at 1 fps and 30 at
  3 FPS with interpolated time embeddings; 5 images in training and 32 shots at test; p_next = 1/2; image-first ordering at
  inference (App. A.3.2); M3W 43.3M documents; ALIGN 1.8B, LTIP 312M, VTP 27M; 16 benchmarks, six tasks with 32 examples, 57.8
  vs 54.4, 67.6 vs 80.2; 1536 chips for 15 days; six vs seven handled correctly (Table 1 caption says seven; Sec. 3.1 and Fig. 2
  say six). Gate toy reproduces: KL 0.000 at step 0; no-gate flips cat (56%) to car (54%); trained ends dog 96%. The
  zero-gradient argument is correct and labelled as derived.
- ch-qformer: 188M, BERT-base init, random cross-attention, every other block, 32 x 768, 257 x 1024, EVA-CLIP ViT-g/14,
  second-last layer (Sec. 3.1, 3.4); the three masks (mask JS reproduces 169 / 89 / 119 open cells), max-over-queries ITC,
  in-batch negatives, averaged ITM logits, [DEC], UniLM (Sec. 3.2, Fig. 2); FC layer, soft visual prompts, forgetting quote,
  LM vs prefix LM (Sec. 3.3); 129M, 10 captions, CLIP ranking, top two, 250k / 80k steps, 16-A100 machine, <6 and <3 days,
  224 x 224 (Sec. 3.4); Table 2: 65.0 vs 56.3, 50.6 vs 45.9, 108M / 12.1B, 10.2B / 80B, trainable range 103M-108M; the
  OK-VQA explanation is the paper's hypothesis. 54x and 94x arithmetic correct (10.2B / 188M = 54.3; 10.2B / 108M = 94.4), and
  the page labels its own reading of the discrepancy.
- ch-llava: Table 1 contexts and responses (abridged faithfully), "not used to prompt GPT", seed-example quote, 158K / 58K / 23K
  / 77K; LLaVA-Bench COCO ablation 85.1 / 73.8 / 21.5; ScienceQA 90.92, 92.53, 91.68, last layer 89.96 and the hypothesis; Eq. 1
  exact; "simple linear layer". LLaVA-1.5: MLP inspired by self-supervised learning with no activation named; both formatting
  prompts; the yellow-shirt example is after stage-1 alignment only (Table 1b caption); all ten scaling-table rows; 558K, 665K,
  1.2M, ~1 day on one 8-A100 node, ~6 h + ~20 h; 11 benchmarks; "hundreds of millions" matches the intro's wording. 256 and 576
  are correctly marked as computed (neither paper states them), and the BLIP-2 224 x 224 grid label is right.

## Bare numbers (check 3)

From build/bare-numbers.txt and my own pass over these chapters' visible text, the sourced numbers without a claim are items
11-13 (80; the repeated 10B; Flamingo 64 / BLIP-2 32 in the connectors caption) plus the caption's "t = 10". Everything else
bare is structure, labels or arithmetic: section, table, figure and equation numbers (all correct for the compiled papers), model
names (ViT-L/14, 336-pixel, B/16, 7B, 13B), UI labels (Train hot (tau = 1), Step 0), repeats of the wrapped -10, and toy seeds.
The repeated "-10" and the model sizes in the LLaVA-1.5 table rows ("7B unless noted", "+ 13B") can stay bare. No bare number
in these chapters is wrong. Tooling gap: check-page skips whole data-toy / data-computed figures, captions included (item 13).

## Toy labelling (check 4)

viz-contrastive, viz-siglip, viz-capfilt and viz-gate carry the Toy badge and data-toy, and their captions say what is toy.
viz-capfilt keeps its only real numbers in the labelled green box. viz-siglip-batch, viz-tokens, viz-resampler and viz-connectors
compute what they claim from the stated rules (D^2, N = HW/P^2, 64 out regardless of input, (side/14)^2). viz-capfilt-med and
viz-qformer are Sourced and their structure and numbers are the papers'. Problems: the toy behaviour described in prose or
caption for viz-contrastive (item 1) and viz-siglip (item 2), and the missing Toy badges in item 24.

## Cross-chapter consistency (check 5)

Consistent across chapters: CLIP batch 32,768; ALIGN 1.8B (ch-shared-space and ch-flamingo); 224 / 336 and 256 / 576 (ch-patches,
ch-llava, ch-training-step); 64 latents and 32 queries; BLIP 14M / 129M = 14M + 115M LAION (ch-bootstrap, ch-qformer); Flamingo-80B
new blocks 10B vs 10.2B trainable (the 194M resampler accounts for the gap); BERT-base (BLIP, Q-Former). Inconsistent: item 4
against ch-bootstrap (ImageNet-initialized ViT) and ch-early-fusion (no encoder); item 6 against the chapter's own MED figure and
Table 3; item 3 against ch-shared-space's hot/cold convention. Minor: ch-qformer mentions "BLIP's momentum queue", which
ch-bootstrap never introduces (it mentions only the momentum encoder); BLIP-2's text supports the phrase, so this is optional.
