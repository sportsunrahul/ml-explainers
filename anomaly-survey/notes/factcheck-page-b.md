# Fact check: page, group B (prose and captions)

Checked 2026-10-10. Scope: `parts/ch-fastflow.html`, `ch-synthetic.html`, `ch-efficientad.html`, `ch-uniad.html`, `ch-winclip.html`, `ch-anomalyclip.html`, `ch-dinov2.html`; `build/f7/ch-speed.tpl.html`, `build/f7/ch-closing.tpl.html` and the labels in `build/f7/gen.py` (regenerated with `python3 -I build/f7/gen.py`); visible text in the 15 viz files listed in the brief. Sources: LaTeX in `sources/<paper>/src`, claim records in `notes/claims.json`. `bash build.sh` prints PASS after the fixes.

Checked and correct (no change): every equation's TeX against the source (FastFlow Eq. 2 and 3; CutPaste Eq. 1 and 2; DRAEM Eq. 2 to 5; EfficientAD inline losses and Alg. 2; UniAD Eq. 1 to 5; WinCLIP Eq. 1, 3, 4, 6; AnomalyCLIP Eq. 2, the local loss and the App. C DPAM weights; AnomalyDINO Eq. 3 and 4; Dinomaly Eq. 3), all equation numbers and versions, every direct quote except the one below, all chapter cross-references, and the frozen/trained/gradient statements in the eight archflow figures. All toy figures carry the toy badge and caption. No non-ASCII punctuation in any file in scope.

## Problems fixed

Severity: high = a sentence was false; medium = right number, wrong or missing variant, protocol or provenance; low = bare number or wording.

| Sev | File | Quoted text (before) | Issue | Fix |
|---|---|---|---|---|
| high | ch-uniad.html | table header "One model per category (own paper)" | UniAD's separate PaDiM column (95.5, per class 99.9 / 92.7 / 91.3 ...) is DRAEM Table 1's PaDiM run, not PaDiM's paper (95.3, WR50-Rd550). CutPaste's 96.1 is CutPaste's 5-model ensemble (per class 98.2 / 81.2 / 98.2 ... = Ensemble column), not the single 3-way model (95.2). | Header now "as printed in UniAD"; added a note with `padim-mvtec-img-auroc-wr50-rd550` and `cutpaste-mvtec-img-auroc-3way`. |
| high | ch-speed.tpl.html (note) | "FastFlow reports only its time on top of the backbone" | False: FastFlow Table 1 also gives FPS including the backbone (3.08 for CaiT); ch-fastflow says so. | Rewritten with `fastflow-fps-cait` and `fastflow-adtime-cait`, batch size not stated. |
| high | ch-dinov2.html | "Chapter 15 puts that next to the other families." | Dinomaly is not on the speed chart, and 58.1 im/s is batched throughput (Table 4 caption), not latency. | Says it is throughput with batches, so it is not on the latency chart. |
| high | ch-closing.tpl.html (deeper) | AnomalyCLIP "Sec. 3.3, Eq. (2) and (3) (the glocal loss)" | Eq. 3 is the deep-text-token recursion; the local loss is unnumbered. | "Eq. (2) and the unnumbered local loss (the glocal loss), and Eq. (3) (the deep text tokens)". |
| high | ch-efficientad.html | "Both teacher and student are a patch description network (PDN) of only four convolutional layers." | Four layers is EfficientAD-S (Table 6); the M PDN has six (Table 7). | Scoped to EfficientAD-S; M "uses a deeper, wider PDN". |
| high | ch-speed.tpl.html (claim card) | "The authors re-ran ten methods" | Eight re-run baselines plus EfficientAD-S and -M (own runs). | "timed EfficientAD-S, EfficientAD-M and every baseline, which they re-ran themselves". |
| high | viz-draemabl.js, ch-synthetic.html aria-label | "Fakes: no texture augmentation" (97.4) | DRAEM_no_aug removes augmentation AND opacity randomization (iccv.tex:624); augmentation alone off is DRAEM_beta, 97.9. | Label "Fakes: no aug., no random opacity"; aria-label spelled out. Screenshot checked at 900 px and 375 px. |
| medium | viz-draemabl.js, ch-synthetic.html | "Plain segmenter on the same fakes" (92.8) | The row is Bozic et al.'s supervised network re-trained on the fakes, not a plain DRAEM segmenter (that is the 93.9 Disc. row). | "Bozic et al. segmenter, same fakes"; readout and aria-label updated. |
| medium | ch-synthetic.html | quote "CutPaste is still not a perfect simulation of real defects and learning on it to find irregularity generalizes well on unseen anomalies." | Verbatim quote silently dropped the source's "(1)" and "(2)". | Split into two exact quoted fragments. |
| medium | ch-synthetic.html | "In the same table PaDiM has the higher pixel AUROC, 97.4, but only 55.0" | These are DRAEM's own PaDiM run (claim factcheck note). | "DRAEM's own run of PaDiM". |
| medium | ch-efficientad.html | "the hard feature loss alone adds + 1.0 point" | Table 4 is cumulative: +1.0 is on top of map normalization (isolated removal in Table 5 costs 0.7). | "cumulative ablation ... added after the map normalization, the hard feature loss gains + 1.0". |
| medium | ch-efficientad.html | "On an RTX A6000 at batch size 1, EfficientAD-S takes 2.2 ms" | Precision missing: all networks timed in float16 (App. E). | Added "with every network in float16". |
| medium | ch-uniad.html | "(Table 1, MVTec AD image AUROC, 224 pixels)" | 224 is stated only for UniAD's own model; re-run resolution not stated (claim factcheck). | Split: UniAD at 224; re-run resolution not stated. |
| medium | ch-uniad.html | "One shared query lifts it to 87.6" | Table 5a row is one query in the first decoder layer only (vanilla transformer), not a shared query. | Reworded. |
| medium | ch-uniad.html | "Training it too drops image AUROC to 65.9: the backbone learns features that are easy to rebuild" | Paper says "We speculate" (supp. 2.ablations.tex:177). | "The authors suspect ...". |
| medium | ch-dinov2.html (claim card) | "96.6 ... at 672 pixels ... About 60 ms per image" | 60 ms is the 448-pixel default, not the 672 setting. | "At the default 448 pixels it takes about 60 ms". |
| medium | ch-dinov2.html (claim card) | "earlier multi-class methods (rows copied from the ADer benchmark)" | Dinomaly says "most" rows are from ADer; UniAD, DiAD, MambaAD rows do not match ADer (claim factcheck; the figure caption already says so). | Aligned with the caption. |
| medium | ch-dinov2.html (caption) | "AnomalyDINO does not say where its 8- and 16-image APRIL-GAN rows come from" | AnomalyDINO line 383 says all other rows come from the original publications; APRIL-GAN reports only up to 4 shots. | "AnomalyDINO says all other rows come from the original papers, but the APRIL-GAN paper stops at 4 images, so the source ... is unclear"; "re-ran nothing" softened to "by its own account". |
| medium | ch-anomalyclip.html | "A token's value vector is most similar to itself, so a^vv_ii is the largest weight in its row" | Not true in general for dot products; the paper argues it from V not taking part in the original attention. | Attributed to the paper, "tends to dominate". |
| medium | ch-anomalyclip.html | "in a separate path that feeds only the patch tokens; the class token keeps the original attention" | Read from Fig. 2, not stated in the text (notes/papers/anomalyclip.md). | Labelled "our reading of that figure". |
| medium | ch-speed.tpl.html | "On an RTX A6000, PatchCore-Ens takes 147.6 ms ..." | Baselines are EfficientAD re-runs in float16; provenance missing. | Named the source, float16, one image at a time. |
| medium | ch-speed.tpl.html | "a CLIP ViT-L/14 image encoder alone takes 130.26 ms" | Precision/runtime missing (TensorRT FP16). | Added. |
| medium | ch-speed.tpl.html | RD "from 166.60 ms to 19.68 ms" | The baseline is PyTorch FP32, so the gain mixes runtime and precision. | "in PyTorch FP32". |
| medium | ch-speed.tpl.html | EdgeZSAD "zero-shot ... 91.6" | Claim factcheck asks for the Real-IAD-DR category-name overlap caveat. | One sentence added, marked our reading. |
| medium | ch-speed.tpl.html (caption), gen.py UQ label | "the latency (one image at a time) and the image AUROC" | Several sources do not state batch size; the Xavier NX rows plot a "mean AUROC" of unstated level on the image-AUROC axis. | Caption says both; UQ source label says the level is not stated. |
| medium | ch-speed.tpl.html (note) | "AnomalyDINO reports 60 ms per image" | Setting missing. | "with one good image at 448 pixels". |
| medium | ch-closing.tpl.html | "best AU-PRO of 58.7%" | That is AU-PRO at FPR 0.30; MVTec AD 2's own metric uses 0.05 (the deeper section says so). | Limit named. |
| medium | ch-closing.tpl.html | "prefer EfficientAD: in the EfficientAD paper's re-runs it scores 90.7 ... against PatchCore's 80.3" | 90.7 is EfficientAD-M's own run; only PatchCore is re-run. | Reworded with variant and provenance. |
| medium | ch-closing.tpl.html | "On a Jetson Orin Nano, EfficientAD takes 34.7 ms and PatchCore ... 119.0 ms" | These are the continual-learning variants EfficientAD* and PatchCore* of one benchmark. | Named as such; EdgeZSAD gets "with TensorRT FP16"; EfficientAD-S guide line gets float16. |
| medium | gen.py | EAD source "(re-runs by the EfficientAD authors)" on EfficientAD-S/M rows | EfficientAD rows are own runs. | "float16, one image at a time; baselines re-run by the EfficientAD authors". |
| medium | gen.py | tree key label "ms, STFPM FP16 on Jetson Xavier NX" | Only the student is FP16; the teacher stays FP32 (claim factcheck). | "STFPM with an FP16 student". |
| low | ch-closing.tpl.html, viz-scoreboard.js | scoreboard caption and sort readout | Rule 5: sorting across rows could read as a ranking. | Caption and readout say rows are not a controlled comparison. |
| low | ch-winclip.html | "From the CLIP chapter you know" | There is no CLIP chapter on the page. | "You know how CLIP classifies". |
| low | ch-dinov2.html | Dinomaly Eq. 3 shows phi(x) = elu(x) + 1 | The paper gives elu+1 only as the usual choice. | Eqno says so. |
| low | ch-fastflow.html | "FastFlow keeps 99.2 image AUROC" | Variant missing (CaiT). | "FastFlow with CaiT". |
| low | several | bare sourced numbers | Rule 4. | Wrapped: 448/256 (FastFlow results), 256 x 256 (DRAEM caption), 7/14/272 (UniAD toy caption), ViT-B/16+ and 240 and 518 (AnomalyCLIP), 448/672/392 (DINOv2). |

## Not fixed (and why)

- `viz-draemflow.js` readout "Half of the training samples stay clean (code-only)": true of the official code, but no claim exists and claims are out of scope. Low.
- `ch-synthetic.html` "divided by 10 after 400 and 600 epochs": the factor 0.1 has no claim id. Low.
- `ch-winclip.html` "K good images (1, 2 or 4)" and the toy caption's "15 by 15 patch grid": protocol settings and arithmetic from the 225-token claim; no claim id fits. Low.
- `ch-efficientad.html` figure caption "Alg. 1 confirms the 384 by 64 by 64 output": a shape from Alg. 1 with no claim id. Low.
- `ch-fastflow.html` "The same table credits CFLOW-AD with 98.6": the claim cites Table 2, but Table 1 prints the same 98.6 (WRN50 row), so the sentence is true either way.
- `ch-anomalyclip.html` deep tokens "first 9 text layers": App. A says the first 9 layers, but Sec. 3.3 says replacement starts "from the bottom (the second layer)". The paper is inconsistent; the page follows App. A. Flag only.
- The multiplication sign in JS shape strings (for example "3x256x256" drawn with a times glyph) is a display glyph inside string literals, not prose punctuation; left as is.
- Tree branch one-liners come from `notes/branches-*.json`, outside this scope; not reviewed.
- During the run another agent edited `ch-spade.html` and `ch-padim.html`; one intermediate build failed on their uncertain claims, and the final build passes.
