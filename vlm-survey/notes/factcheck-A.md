# Fact-check A: vit, clip, align, siglip, blip, blip2, flamingo, llava

Method: every claim in notes/claims/{vit,clip,align,siglip,blip,blip2,flamingo,llava}.json was compared with the LaTeX in
sources/<key>/src/ (SigLIP: pdftotext + page text of paper.pdf). Numbers were checked cell by cell against the LaTeX tables
or PDF tables; section numbers were checked against the compiled section headings; "para N" was checked by counting
blank-line-separated paragraphs after the section heading. Claims files were not edited.

Headline: no numeric or textual VALUE was found wrong (all 219 values match the source), except one wording issue
(align-abstract-size). The real defects are (1) llava.json has scrambled field names in all 26 entries, and (2) 22 loc
strings point at the wrong section/table/paragraph.

## A. Structural defect (blocks any consumer of the file)

| id | problem | what the file has vs what it should have |
|---|---|---|
| llava.json, all 26 ids | Field names shifted. `paper` holds the loc string, `version` holds "LLaVA", `loc` holds "v2". Every other claims file has paper="<name>", version="vN", loc="<location>". | e.g. llava-instruct-total: "paper": "Sec. 3 last para", "version": "LLaVA", "loc": "v2" -> should be "paper": "LLaVA", "version": "v2", "loc": "Sec. 3 last para". |

## B. WRONG (value or wording not what the source says)

| id | problem | source says | location |
|---|---|---|---|
| align-abstract-size | Value "over one billion noisy image alt-text pairs" is the Sec. 1 wording, not the abstract's. loc says Abstract. | Abstract: "a noisy dataset of over one billion image alt-text pairs, obtained without expensive filtering or post-processing steps in the Conceptual Captions dataset". Sec. 1 (line 131 of align.tex): "over one billion noisy image alt-text pairs". | Abstract (align.tex l.98) vs Sec. 1 (l.131) |
| llava-connector | Value says "single linear layer"; not verbatim. | "We consider a simple linear layer to connect image features into the word embedding space." (no "single") | Sec. 4.1 (approach.tex l.101) |

## C. MISLOCATED (value correct, loc wrong)

Section/table level (these would send a reader to the wrong place):

| id | claimed loc | actual loc |
|---|---|---|
| clip-l14-res | Table 19 (CLIP-ViT hyperparameters) | Table 20 "CLIP-ViT hyperparameters" (Table 19 is CLIP-ResNet). Row: ViT-L/14 224, ViT-L/14-336px 336. |
| clip-zs-imagenet | Table 1; Sec. 3.1.2 | Table 1; Sec. 3.1.3 "Initial Comparison to Visual N-Grams" (3.1.2 is "Using CLIP for Zero-Shot Transfer") |
| clip-zs-top5 | Sec. 3.1.2 | Sec. 3.1.3 ("95% top-5 accuracy, matching Inception-V4") |
| clip-matches-rn50 | Sec. 3.1.2; abstract | Sec. 3.1.3; abstract |
| clip-rn50-sup-number | Sec. 3.1.2 | Sec. 3.1.3 (value "not stated" is right; still uncertain=true) |
| clip-wins-16-27 | Sec. 3.1.3 / Fig. 5 | Sec. 3.1.5 "Analysis of Zero-Shot CLIP Performance"; Fig. 5 caption and text "wins on 16 of the 27 datasets" |
| clip-ensemble-gain | Sec. 3.1.4 para 4 | Sec. 3.1.4 para 5 (para 4 is the task-specific "a type of pet" paragraph) |
| clip-prompt-total | Sec. 3.1.4 para 4 | Sec. 3.1.4 para 5 |
| align-loss-combo | Sec. 3 (before Eq. 1) | Sec. 4.1 "Pre-training on Noisy Image-Text Pairs" (Sec. 3 is the dataset) |
| align-temp-learned | Sec. 3 after Eq. 2 | Sec. 4.1, paragraph after Eq. 2 |
| align-temp-converged | Sec. 8.1 (rows 7-9) | Sec. 6.1 "Model Architectures", Table 8 rows 7-9 ("converged to about 1/64"); Sec. 8 is Multilingual ALIGN |
| align-scale-vs-noise | Sec. 8.2 | Sec. 6.2 "Pre-training Datasets", Table 10 text ("ALIGN data outperforms Conceptual Captions with only 4x size") |
| align-frozen-imagenet | Table 3 | Table 5 "ImageNet classification results" (Table 3 is CxC Spearman), Sec. 5.3; row ALIGN (EfficientNet-L2) 85.5 frozen |

Paragraph-index only (minor; section is right, paragraph number is off by one or more):

| id | claimed loc | actual loc |
|---|---|---|
| vit-pos-emb | Sec. 3.1 para 4 | Sec. 3.1 para 3 ("Position embeddings are added ... standard learnable 1D") |
| vit-imagenet-only | Sec. 1 para 3 | Sec. 1 para 4 ("When trained on mid-sized datasets ... few percentage points below ResNets") |
| vit-trumps | Sec. 1 para 3 | Sec. 1 para 5 ("large scale training trumps inductive bias") |
| blip2-queries | Sec. 3.1 para 3 | Sec. 3.1 para 2 ("In our experiments, we use 32 queries ... 768") |
| blip2-image-feat-size | Sec. 3.1 para 3 | Sec. 3.1 para 2 (257x1024 vs 32x768) |
| blip2-qformer-init | Sec. 3.1 para 2 | Sec. 3.1 para 1 |
| blip2-qformer-params | Sec. 3.1 para 2; Table 1 | Sec. 3.1 para 1; Table 1 |
| blip2-xattn-freq | Sec. 3.1 para 2 | Sec. 3.1 para 1 |
| blip2-stage2-loss | Sec. 3.3 para 3 | Sec. 3.3 para 2 ("We experiment with two types of LLMs ...") |
| flamingo-mask-train-images | Sec. 2.3 para 2 | Sec. 2.3 has a single paragraph; the 5-vs-32 sentence is its last-but-one sentence |

Note on the para-index rows: the BLIP-2 Sec. 3.1 numbers are all +1 (as if the section intro were counted); if that is the
intended convention, the sibling claim blip2-stage2-connection ("Sec. 3.3 para 1") is inconsistent with it. Pick one convention.

## D. Should be uncertain=true (source does not cleanly support the flag "false")

| id | why | source |
|---|---|---|
| flamingo-headline-six | The source contradicts itself: Sec. 3.1, Fig. 2 caption and intro say six tasks; the Table 1 caption says "seven tasks". Table data supports six (OKVQA 57.8>54.4, MSVDQA 52.3>47.9, Flickr30K 75.4>67.4, iVQA 45.3>35.4, STAR 42.2>36.7, NextQA 33.5>25.2). The claim text already records the discrepancy, so this is borderline; uncertain=true (or a "discrepancy" marker) would stop a page from printing either number unqualified. | Table 1 caption vs Sec. 3.1 |

Info only (not an error): siglip-clip-cost is uncertain=true although the source states it ("approx. 2500 TPUv3-days for 72.6% ... reported in [30]", Sec. 4.5). The flag is harmless but unnecessary; the only caveat is that it is a second-hand number from FLIP's paper. siglip-t1-siglit-b8 (79.8 table cell vs 79.7 caption/text) is correctly uncertain: confirmed in Table 1 row 1 vs the caption and Sec. 1/4.4.

## E. Errors in notes/papers/*.md

1. siglip.md, "Sigmoid loss" gloss: says the printed loss "equals log sigmoid(u), u = z_ij (t x_i.y_j + b)". The printed equation (Sec. 3.2, unnumbered) is
   `log 1/(1 + e^{z_ij(-t x_i.y_j + b)})`, which equals log sigmoid(z_ij (t x_i.y_j - b)), i.e. the opposite sign on b from what the gloss states and from Algorithm 1 (`logits = dot(zimg, ztxt.T) * t + b`, `log_sigmoid(labels * logits)`). The TeX transcription is symbol-for-symbol correct; the gloss silently "fixes" a sign inconsistency inside the paper and should instead flag it (equation and Algorithm 1 disagree on the sign of b; with b initialised at -10 only the Algorithm 1 form makes sense as a prior on negatives).
2. siglip.md, Algorithm 1 heading: "(the v4 fix of an earlier bug)". The paper only says (Acknowledgements) that Ross Wightman spotted a pseudocode mistake in the first version and that typos in v2/v3 made t vs t' confusing. That v4's Algorithm 1 is the fix is an inference, not a statement.
3. clip.md, "Temperature (Sec. 2.5 Training ...); Sec. 3.1.1: pre-training step viewed as a proxy dataset": the "1 example per class and has 32,768 total classes" sentence is in Sec. 3.1.2 "Using CLIP for Zero-Shot Transfer", not 3.1.1.
4. clip.md, "Zero-shot ImageNet (Sec. 3.1.2 + Table 1 ...)": the Table 1 discussion (76.2, 95% top-5, matches ResNet-50) is Sec. 3.1.3.
5. clip.md, Mechanism note "the prose (Sec. 2.5 zero-shot) says cosine similarity is 'scaled by a temperature parameter tau'": that sentence is in Sec. 3.1.2.
6. clip.md, "224 vs 336 (Sec. 2.5; Table 19 appendix)": the ViT hyperparameter table is Table 20.
7. align.md, Temperature: "Ablation (Sec. 8.1)": it is Sec. 6.1 (Table 8). Dataset: "Ablation (Table 8): ALIGN data worse than CC at 3M ... 4x size": the data-size ablation is Table 10 in Sec. 6.2; Table 8 is the architecture/temperature ablation.
8. align.md, Loss header "(Sec. 3 / 'Pre-training')": the loss is in Sec. 4.1, not Sec. 3.
9. flamingo.md, Sec. 4 (Table 5) last bullet: "Text of App. B.1.1 gives 1.4B / 1.8B / 10B for 'additional learned parameters'; these match table xattn-dense + resampler". True for 3B (1.2B+194M) and 9B (1.6B+194M) but not 80B: the text says 10B, and table xattn-dense 10B + 194M would be 10.2B. So the explanation holds for two of three sizes only. (The same file's outline-assertion list says the 80B trainable count is "about 10.2B (derived)", which is fine; only the generalisation is off.)
10. vit.md: all equations (1)-(4), Table 1, Table 2/5 numbers, token counts verified with no error. Minor imprecision only: "Params are per variant, independent of patch size" is not stated in the paper (Table 1 gives no patch size; E and E_pos do depend on P and N).
11. blip.md, blip2.md, llava.md: no errors found. All equations in llava.md (Eq. 1-3) match approach.tex; blip.md/blip2.md have no equations (prose only, as the notes say). blip2.md correctly labels the "trainable in stage 2" statement and the 94x ratio as inference. llava.md correctly labels the 256-token count and the box coordinate order as not in the paper.

## F. Verified with no finding (for the record)

All numeric values in: vit (28 claims, Tables 1/2/5, Sec. 4.1/4.3/4.4), clip (Table 1, Table 20, prompt gains, dataset stats, pseudocode lines checked against the rendered PDF text), align (Eq. 1/2 symbol-for-symbol, filters, Tables 4/5/8/10), siglip (Table 1, Table 2, Table 4, Table 5, Table 8 cell by cell; Sec. 3.2/3.3 equations and Algorithm 1 vs PDF), blip (Tables med/capfilt/generate/independent cell by cell), blip2 (Tables 1/2, Sec. 3.1-3.4), flamingo (Tables 1-5, 10, Fig. 4/5 pseudocode, Eq. 1/2, appendix B.1), llava (Sec. 3-5 numbers, Tables). Versions (v1/v2/v3/v4) match the extract-brief keys for all files except the llava field shift.

## Counts

- Claims checked: 219 (vit 28, clip 25, align 19, siglip 31, blip 22, blip2 27, flamingo 41, llava 26); notes files read: 8
- Wrong (value/wording): 2 (align-abstract-size, llava-connector), plus 1 file-level defect (llava.json field shift affecting all 26 entries)
- Mislocated: 23 (13 section/table level: clip 8, align 5; 10 paragraph-index only: vit 3, blip2 6, flamingo 1)
- Should-be-uncertain: 1 (flamingo-headline-six, borderline)
- Notes errors: 9 substantive (items E1-E9), 2 minor/no-error items
