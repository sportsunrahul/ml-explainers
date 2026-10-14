# DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR

Liu, Li, Zhang, Yang, Qi, Su, Zhu, Zhang. ICLR 2022. arXiv 2201.12329v4 (v4 dated 30 Mar 2022 on the PDF stamp).
Source read: `sources/dab-detr/tex/DAB-DETR.tex` plus every `\input` file; equation, table, and figure numbers checked against `paper.pdf` (v4).
Official code: github.com/SlongLiu/DAB-DETR (now redirects to IDEA-Research/DAB-DETR). Anything marked "[repo]" comes from the code or README, not the paper.

---

## Delta from predecessors

**What an object query actually is (the paper's framing).** In DETR every decoder query has two parts:
- a *content part*: the decoder embedding (the self-attention output; initialized to 0 in the first layer),
- a *positional part*: in DETR, this is the learnable query vector (the "object query").

Keys in cross-attention also have two parts: a content part (encoded image feature) and a positional part (sinusoidal positional embedding of the pixel location). So cross-attention similarity = content similarity (pool semantically related features) + positional similarity (constrain pooling to a region around the query position). Footnote in Sec. 1 notes the learnable queries in DETR are "only for the positional part", and credits Conditional DETR with related discussion. [claim dab-query-two-parts]

**Why DETR queries train slowly (Sec. 3, "Why a Positional Prior could Speedup Training?").**
1. Following Sun et al. (TSP), the slow convergence is blamed on decoder cross-attention. Comparing encoder self-attention with decoder cross-attention (Fig. 2), keys and values are the same kind of thing; the only difference is the queries. Decoder embeddings start at 0 and are projected into the image-feature space after the first cross-attention, after which they behave much like encoder features. So the root cause is "likely due to the learnable queries".
2. Two hypotheses: (a) queries are hard to optimize; (b) the positional information in learned queries is not encoded the same way as the sinusoidal encoding used for image features.
3. Test of (a): reuse well-learned DETR queries, freeze them, train everything else. Fixed queries "only slightly improve the convergence in very early epochs, e.g. the first 25 epochs" (Fig. 3a). So optimization is "likely not the key concern". [claim dab-fixq-25ep]
4. Test of (b): visualize positional attention between learned queries and image positional embeddings (Fig. 4a). DETR's maps show **multiple modes** (two or more concentration centers) and **nearly uniform / unconcentrated** weights (regions too large or too small). The authors *conjecture* the multi-mode property is "likely the root cause" of slow training.
5. Verification: swap DETR's query formulation for dynamic anchor boxes ("DETR+DAB"), no other change (no 300 queries, no focal loss). Fig. 3b: better AP and lower train/test loss. Figure 3 curves are 3 runs each, mean plus 95% CI. [claim dab-fig3-runs]

**Versus Conditional DETR (the direct competitor).** Conditional DETR uses explicit positional embeddings of a 2D reference point as positional queries, producing Gaussian-like attention maps (Fig. 4b), but "it cannot adapt to objects of different scales". Its per-layer queries are still derived from a high-dimensional learned query shared across layers, so it cannot cleanly do layer-by-layer refinement ("it is unclear how to convert an updated anchor back to a high-dimensional query embedding", Sec. 4.3). DAB-DETR borrows two pieces from it: the concatenated (decoupled) content/position cross-attention query and key, and the conditional spatial query scaling MLP (Eq. 4).

**Versus Anchor DETR (concurrent work).** Anchor DETR also learns anchor *points* directly and updates them layer by layer, but ignores anchor width and height, so cannot modulate attention by scale. Its positional queries are high-dimensional and passed to self-attention in all layers "without any adaptation", which the paper calls possibly sub-optimal since self-attention cannot use the refined points (Appendix B). DAB-DETR also borrows Anchor DETR's multiple pattern embeddings (3 patterns) for its starred models.

**Versus Deformable DETR.** Deformable DETR already has 4D boxes updated per layer ("iterative bounding box refinement") and uses w, h to scale its sampling area, but that is tightly coupled to deformable attention, so "nontrivial to apply it to general Transformer decoder-based DETR models". Its position queries are also high-dimensional and passed to self- and cross-attention in all layers without adaptation (Appendix B). DAB-DETR brings the 4D-box idea to *standard dense* cross-attention. Table 1 flags: Deformable DETR = reference anchors 4D, dynamic anchors yes, size-modulated yes, standard attention no, learns anchors no.

**Table 1 summary (Learn Anchors? / Reference Anchors / Dynamic Anchors / Standard Attention / Size-Modulated Attention / Update Learned Spatial Queries?).** [claim dab-table1]
- DETR: No / No / - / yes / - / -
- Deformable DETR: No / 4D / yes / - / yes / -
- SMCA: No / 4D / - / yes / yes / -
- Conditional DETR: No / 2D / - / yes / - / -
- Anchor DETR: 2D / 2D / yes / - / - / -
- Sparse RCNN: 4D / 4D / yes / - / - / -  (not DETR-like; listed for similar anchor formulation)
- DAB-DETR: 4D / 4D / yes / yes / yes / yes

Other related work mentioned briefly: SMCA (pre-defined Gaussian maps around reference points), Efficient DETR (dense module picks top-K positions as queries), TSP (encoder-only), Sparse R-CNN (learns boxes but drops Transformer, uses hard ROI align), Dynamic DETR (ROI pooling, local only, loses global context; Appendix B).

---

## The one key idea

**A query's positional part should literally be a box.** Replace DETR's learned high-dimensional positional query with an explicit, learnable 4D anchor box A_q = (x, y, w, h). Derive the positional query from it each layer (sinusoidal encoding + MLP), use (x, y) to center the positional attention and (w, h) to stretch it into an oval matched to object size, and refine the box layer by layer with predicted deltas. The paper's own summary: queries in DETR "can be implemented as performing soft ROI pooling layer-by-layer in a cascade manner". Title thesis: "anchor boxes are better queries for DETR"; the paper's stated new perspective is that "all information contained in queries are box coordinates".

---

## Mechanism step by step

Setting: CNN backbone -> 6 Transformer encoders -> 6 Transformer decoders -> shared box and class heads; Hungarian matching loss as in DETR [claim dab-enc-dec-layers]. Only the decoder changes. Fig. 5 is the framework figure.

"Dual queries" enter the decoder: positional queries (anchor boxes) and content queries (decoder embeddings). Both are updated layer by layer "to get close to the target ground-truth objects gradually".

### 1. Anchor -> positional query (Eq. 1, 2)
- Each of the 300 queries owns a learnable anchor A_q = (x_q, y_q, w_q, h_q) [claim dab-300-queries].
- Encode each of the four floats separately with a sinusoidal PE, concatenate (Eq. 2). PE maps R -> R^{D/2}, so the concat is 2D-dimensional.
- An MLP (two linear+ReLU submodules, reduction at the first linear layer) maps R^{2D} -> R^D to give P_q (Eq. 1). The MLP's parameters are **shared across all layers**.
- [repo] D = hidden_dim default 256 [claim dab-repo-d256]; sine embedding multiplies coordinates by 2*pi before encoding; concat order in code is (y, x, w, h) [claim dab-repo-sine-dims].

### 2. Self-attention among queries (Eq. 3)
- Q = K = C_q + P_q, V = C_q. Positional query is *added* to content, like DETR.
- Because P_q is recomputed from the current anchor, self-attention sees the refined boxes each layer (the paper's contrast with Anchor DETR and Deformable DETR, whose high-dimensional positional queries are passed to every layer unchanged).

### 3. Cross-attention to image features (Eq. 4)
- Following Conditional DETR, content and position are **concatenated**, not added, so the dot product decouples into content-content plus position-position terms.
- Query positional half = PE(x_q, y_q) (only the 2D center) times, elementwise, a scale vector MLP^(csq)(C_q) conditioned on content (Conditional DETR's "conditional spatial query").
- Key = Cat(F_{x,y}, PE(x, y)), value = F_{x,y}. Both positional halves come from 2D coordinates with the same encoder, so they are directly comparable.
- [repo] In code the csq scale is set to 1 for the first decoder layer and only applied from layer 2 on [claim dab-repo-csq-layer0].

### 4. Width and height modulated positional attention (Eq. 5, 6, 7)
- The plain positional attention logit is PE(x).PE(x_ref) + PE(y).PE(y_ref), over sqrt(D) (Eq. 5). With a sinusoidal PE this acts like an isotropic Gaussian-like bump of fixed size around the anchor center.
- DAB-DETR scales the x term by w_{q,ref}/w_q and the y term by h_{q,ref}/h_q, **before softmax** (Eq. 6). A wider anchor (larger w_q) shrinks the x term, flattening the attention along x; a narrow one sharpens it. Result: an oval prior matched to the box's aspect ratio and size (Fig. 4c, Fig. 6).
- w_{q,ref}, h_{q,ref} = sigmoid(MLP(C_q)) (Eq. 7): a content-dependent reference size, so the modulation is relative.
- [repo] Implemented by multiplying the x-half and y-half of the query sine embedding by refW/w and refH/h respectively (flag `modulate_hw_attn=True`).

### 5. Layer-wise anchor update (Sec. 4.3)
- After each decoder layer a prediction head outputs relative offsets (Delta x, Delta y, Delta w, Delta h); the anchor is updated and fed to the next layer, "following the previous practice" of Deformable DETR and Anchor DETR.
- All prediction heads in different layers **share the same parameters** (Sec. 4.3; also Appendix A for both box and class heads).
- The paper does **not** write the update formula. [repo] Code: `new = sigmoid(Delta + inverse_sigmoid(old))`, i.e. the delta is added in logit space; the new anchor is `.detach()`-ed before being passed to the next layer (gradients do not flow through the anchor chain across layers). [claim dab-repo-update-rule]
- Visual evidence: Fig. 11 (Appendix H) compares layer-by-layer boxes of DAB-DETR vs Conditional DETR; Conditional DETR boxes "have larger variances" and some later-layer boxes are worse than earlier ones (yellow shading), attributed to high-dimensional queries shared across layers making updates "not stable". The LaTeX source also contains an unused figure file (`image_iter.tex`, not `\input` in v4) showing initial anchors / after layer 1 / after last layer with a R50 backbone; images `layer_0.png` .. `layer_6.png` and `images/iter_boxes/img_*_layer_{0,1,6}.png` exist in `tex/iclr2022/images/` and could be reused as reference for the scrubber.
- Appendix D / Fig. 10 (learned initial anchors): 2D-only anchor points spread evenly over the image; with 4D boxes the centers "seem to distribute randomly" (possibly because centers are coupled with sizes); most learned boxes are medium size with no particular pattern.
- Appendix G: freezing the first-layer x, y at random initialization (still updated in later layers) consistently helps slightly; conjectured to reduce overfitting.

### 6. Temperature tuning (Eq. 8)
- Sinusoidal PE uses a temperature T. Transformers hard-code T = 10000, designed for integer word positions; DETR coordinates are floats in [0, 1], so a different T is "highly desired". Larger T -> flatter positional attention (Fig. 7).
- Chosen empirically: T = 20 in all models [claim dab-T20]. The paper calls this "overlooked in all prior works".
- [repo] In the current code, T = 20 is the default for the *image key* positional encoding (`pe_temperatureH/W=20`), while the query-side sine embedding function hard-codes 10000. The paper does not say which side(s) T = 20 applies to. [claim dab-repo-temperature-split]

---

## Equations (v4 numbering, LaTeX copied from source)

**(1)**
```latex
P_q = \text{MLP}(\text{PE}(A_q)),
```
P_q: positional query for query q, in R^D. A_q = (x_q, y_q, w_q, h_q): the q-th anchor box. PE: sinusoidal encoding of floats. MLP: shared across all decoder layers, R^{2D} -> R^D.

**(2)**
```latex
\text{PE}(A_q) = \text{PE}(x_q, y_q, w_q, h_q) = \text{Cat}(\text{PE}(x_q), \text{PE}(y_q), \text{PE}(w_q), \text{PE}(h_q)).
```
Each coordinate encoded separately to D/2 dims and concatenated (Cat) into a 2D-dim vector.

**(3)**
```latex
\text{Self-Attn: } \quad  Q_q=C_q + P_q, \quad K_q=C_q + P_q, \quad V_q=C_q,
```
C_q: content query (decoder embedding) in R^D. Query and key are content plus position; value is content only.

**(4)**
```latex
\begin{aligned}
    \text{Cross-Attn: }\quad &   Q_q=\text{Cat}( C_q, \text{PE}(x_q, y_q)\cdot \text{MLP}^{(\text{csq})}(C_q)),\quad \\
                        &   K_{x,y}=\text{Cat}(F_{x,y}, \text{PE}(x,y)),\quad V_{x,y}=F_{x,y},
\end{aligned}
```
F_{x,y}: image feature at position (x, y). PE(x_q, y_q): encoding of the anchor center only. MLP^(csq): R^D -> R^D, content-conditioned scale vector (Conditional DETR's conditional spatial query). The dot "." is elementwise multiplication. Cat: concatenation, so the attention logit splits into content.content + position.position.

**(5)**
```latex
\text{Attn}((x,y), (x_{\text{ref}}, y_{\text{ref}})) = (\text{PE}(x)\cdot \text{PE}(x_{\text{ref}}) + \text{PE}(y)\cdot \text{PE}(y_{\text{ref}})) / \sqrt{D},
```
(x, y): an image (key) position. (x_ref, y_ref): the anchor center. Here "." is a dot product between encodings. 1/sqrt(D) is standard attention rescaling. This is the positional part of the attention logit.

**(6)** (label `eq:modulated_attn`)
```latex
\text{ModulateAttn}((x,y), (x_{\text{ref}}, y_{\text{ref}})) = (\text{PE}(x)\cdot \text{PE}(x_{\text{ref}}) \frac{w_{q,\text{ref}}}{w_q} + \text{PE}(y)\cdot \text{PE}(y_{\text{ref}}) \frac{h_{q, \text{ref}}}{h_q}  ) / \sqrt{D},
```
w_q, h_q: current anchor width and height. w_{q,ref}, h_{q,ref}: reference width and height from Eq. 7. The x-similarity is scaled by w_ref/w_q and the y-similarity by h_ref/h_q, before softmax. Large w_q -> small multiplier -> softer, wider attention along x.

**(7)**
```latex
w_{q,\text{ref}}, h_{q,\text{ref}} = \sigmoid( \text{MLP}(C_q)).
```
Reference size predicted from the content query, squashed to (0, 1) by sigmoid. (`\sigmoid` is a macro from `math_commands.tex`; renders as sigma.)

**(8)** (label `eq:pe`)
```latex
\text{PE}(x)_{2i}=\sin (\frac{x}{T^{2i/D}}),\quad  \text{PE}(x)_{2i+1}=\cos (\frac{x}{T^{2i/D}}),
```
T: hand-designed temperature (paper uses 20). i indexes frequency pairs; even indices get sin, odd get cos. Larger T -> lower frequencies -> flatter positional attention.

The per-layer anchor update has no numbered equation in the paper (see Mechanism step 5).

---

## Results worth remembering

All COCO val2017, trained on train2017, 50 epochs unless noted. "*" = 3 pattern embeddings (Anchor DETR style).

Headline:
- DAB-DETR-DC5-R50* = **45.7 AP** at 50 epochs, the abstract's headline number [dab-dc5-r50-star-ap].
- DAB-DETR-R50 (no patterns) = 42.2 AP at 50 epochs [dab-r50-ap] vs DETR-R50 = 42.0 AP at 500 epochs [dab-detr-r50-ap] and Conditional DETR-R50 = 40.9 AP at 50 epochs [dab-cdetr-r50-ap]. Same 10x fewer epochs story, now slightly above DETR.
- DAB-DETR-R50* = 42.6 AP [dab-r50-star-ap]; Anchor DETR-R50* = 42.1 AP [dab-anchordetr-r50-ap].
- DC5-R50: DAB-DETR 44.5 [dab-dc5-r50-ap] vs Conditional DETR 43.8 [dab-cdetr-dc5-r50-ap] vs Anchor DETR* 44.2 [dab-anchordetr-dc5-r50-ap] vs DETR (500 ep) 43.3 [dab-detr-dc5-r50-ap]; also above multi-scale Deformable DETR-R50 43.8 [dab-deform-r50-ap] and SMCA-R50 43.7 [dab-smca-r50-ap].
- R101: DAB-DETR 43.5 [dab-r101-ap], DAB-DETR* 44.1 [dab-r101-star-ap]; DC5-R101: 45.8 [dab-dc5-r101-ap], 46.6 [dab-dc5-r101-star-ap].
- Params: DAB-DETR-R50 44M, same as Conditional DETR-R50 44M [dab-r50-params]. GFLOPs: DAB-DETR-R50 94 vs Conditional DETR-R50 90 (as reported by authors; paper says with its own script they are "nearly the same") [dab-r50-gflops].

Ablation, Table 3 (ResNet-50-DC5; row 1 = full model) [dab-abl-*]:
- Full: 45.7.
- No anchor update: 44.0 (so anchor update = +1.7 AP) [dab-abl-no-update].
- No wh-modulated attention: 45.0 [dab-abl-no-modulate].
- 2D anchor point instead of 4D box (with no modulation): 44.0, vs 45.0 for 4D with no modulation [dab-abl-2d].
- No temperature tuning: 44.4 [dab-abl-no-temp].

Temperature sweep, Table 6 (R50, batch 64, no patterns, no modulated attention): T=20 best at 40.1 AP; T=10000 gives 39.5 [dab-temp-*]. T=2 (39.6) vs T=10000 (39.5): similar AP, but small T better on AP_S/AP_M and large T better on AP_L.

Decoder depth, Table 7 (standard R50-DC5 setting; row values match the DC5-R50* model): 2 layers 40.2, 3 43.9, 4 44.9, 5 45.2, 6 45.7 [dab-declayers-2, dab-declayers-3, dab-declayers-4, dab-declayers-5, dab-declayers-6].

DAB-Deformable-DETR, Table 5 (R50 multi-scale, same settings, only query formulation differs): 46.3 -> 46.8 AP (+0.5), "no more than 10 lines of code modified" [dab-deform-plus-os-ap, dab-dabdeform-ap, dab-dabdeform-10loc]. The total loss summed over layers is *higher* for DAB-Deformable-DETR, but the last-layer loss is lower (Fig. 9).

Fixed first-layer x, y (Table 8): +0.1 to +0.7 AP across variants, e.g. DC5-R50* 45.7 -> 45.8, R101* 44.1 -> 44.8 [dab-fixxy-*].

Runtime, Table 9 (single A100): DETR-R50 0.048 s/img, Conditional DETR-R50 0.057, DAB-DETR-R50 0.059 [dab-runtime-*].

Convergence: Appendix K / Fig. 13 (R50 DC5): DAB-DETR (fix x&y variant) converges faster than Conditional DETR "especially on the early epochs". No numbers given in text.

---

## Limitations / what follow-ups fixed

From the paper:
- Single-scale: failure cases (Appendix I, Fig. 12) are dense objects, very small objects, very large objects. Authors say they "will introduce a multi-scale technique". AP_S of DAB-DETR-DC5-R50 (25.3) remains below multi-scale Deformable DETR-R50 (26.4) and TSP-RCNN-R50 (29.7) [dab-dc5-r50-aps, dab-deform-r50-aps, dab-tsp-r50-aps].
- Slight GFLOPs increase over Conditional DETR (as reported), slightly slower runtime (0.059 vs 0.057 s/img).
- The "multi-modal queries cause slow training" claim is stated as a conjecture, supported by Fig. 3b and visualizations, not a proof.
- Temperature T is hand-tuned (T = 20 chosen empirically), not learned.
- Still Hungarian matching with 300 queries; matching instability across epochs is not addressed.

Follow-ups (outside this paper; context only, no numbers taken from them here):
- DN-DETR (arXiv 2203.01305, same group) builds on DAB-DETR and adds denoising training to stabilize bipartite matching; the README lists "DAB-Deformable-DETR (Deformable Encoder Only)" as the "Baseline for DN-DETR" and calls it "a multiscale version of our DAB-DETR" [repo].
- DINO (same group) later combines DAB-style 4D anchor queries, DN-style denoising, and a Deformable DETR backbone (general knowledge; verify against the DINO paper if used).
- README reports a stronger DAB-Deformable-DETR-R50 result than the paper (48.7 for the rendered 'v2' row vs paper 46.8; a 48.1 row exists only inside an HTML comment in the README source), attributed to different loss coefficients [dab-repo-dabdeform-48]. Treat as repo numbers, not paper numbers.

---

## Facts useful for visuals

- Number of queries (anchors): 300 for DAB-DETR (and Conditional/Deformable DETR); DETR uses 100 [dab-300-queries, dab-detr-100-queries].
- Pattern embeddings in starred models: 3 [dab-3-patterns]. [repo] Code repeats each of the 300 anchors per pattern, so effectively 300 x 3 query slots [dab-3-patterns].
- Encoder / decoder layers: 6 / 6 [dab-enc-dec-layers]. So the scrubber has an initial anchor (layer 0 input) plus 6 refinement steps; the final (6th) layer output is the prediction.
- Anchor parameterization: (x, y, w, h), normalized floats in [0, 1] (paper: "floats between 0 and 1"), center plus size. Update adds (Delta x, Delta y, Delta w, Delta h) from a head shared across layers [repo: in logit space].
- Positional attention shape: Gaussian-like bump centered at (x, y); isotropic without modulation, oval with width scaled by w_q/w_ref and height by h_q/h_ref (Eq. 6). Temperature controls overall flatness (larger T = flatter).
- Paper figures to mirror: Fig. 1 (DETR vs Conditional DETR vs DAB-DETR cross-attention block diagram), Fig. 4 (positional attention maps: DETR multi-modal vs Conditional DETR isotropic Gaussian vs DAB-DETR oval), Fig. 5 (framework), Fig. 6 (maps modulated by W, H; source images `H-1-W-1.png`, `H-1-W-3.png`, `H-3-W-1.png`, `H-3-W-3.png`), Fig. 7 (temperatures; `temp-10000.png` exists), Fig. 10 (learned initial anchors), Fig. 11 (layer-wise box updates, DAB vs Conditional DETR).
- D (embedding dim): paper leaves symbolic; [repo] default 256, so PE per coordinate = 128 dims, 4D anchor encoding = 512 dims, MLP 512 -> 256.
- Training: 50 epochs, LR drop by 0.1 after 40 epochs; AdamW, weight decay 1e-4; LR 1e-5 backbone, 1e-4 rest; 16 GPUs x 1 image (though Table 4 lists batch 2/GPU for R50 and R101); focal loss alpha 0.25, gamma 2; matching cost cls 2.0 / L1 5.0 / GIoU 2.0; final loss cls 1.0 / L1 5.0 / GIoU 2.0; PReLU activations [dab-train-*].
- Top-300 predictions by classification logit are kept for evaluation [dab-eval-top300].
