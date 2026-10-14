# Fact check, claims group A

Scope: notes/claims/benchmarks.json, spade-padim.json, patchcore.json, efficientad.json, flows-synthetic.json, branches-a.json (1,439 claims) and map.json (28 claims). Checked 2026-10-10 against sources/*/src LaTeX, pdftotext of paper.pdf, abs.html, and sources/patchcore/code + README. Patches: notes/factcheck-claims-a.json (43 patches; no claims file edited).

Result: no wrong values found. The problems are protocol labels on copied or re-run baseline rows, plus a few untraceable or unverifiable numbers.

## Problems

Severity: medium = a headline number whose protocol or provenance label would mislead the scoreboard or speed chart; low = caveat or unverifiable.

| id | problem | correct value / loc | evidence |
|---|---|---|---|
| efficientad-t1-{gcad,simplenet,st,fastflow,dsr,patchcore,patchcore-ens,ast}-{auroc,aupro} (16) | medium. The protocol says "256x256 input, PDN padding disabled" on baseline rows. That is EfficientAD's own setting. | Value correct. Protocol must give each baseline's config: PatchCore WRN-101, 224x224, 1% coreset, center crop off; PatchCore-Ens 3 backbones, 320x320; FastFlow WRN-50-2, 200 steps; GCAD single 33x33 member. | efficientad main.tex App. B.1-B.7 (lines 925-985) |
| efficientad-t1-patchcore{,-ens}-{latency,throughput}, -t2-patchcore{,-ens}-mvtec, -t12-patchcore{,-ens}-mvtec-aupro (8) | medium. The PatchCore config is missing. The speed chart and scoreboard need it, because the EfficientAD re-run (98.7) differs from PatchCore's own 99.0 at WRN-50/224 with center crop. | Add the App. B.5 config. The same config applies to every PatchCore row in Tables 2, 12, 16 and 17. | App. B.5; Sec. 4 line 400 (76.6% crop disabled) |
| patchcore-t3-student-pro, spade-t3-student, padim-t3-student-pro | medium. The copied "Student" PRO 85.7 is Uninformed Students at single scale p=65. US's own best is multiscale 0.914. | Value correct as printed. Add the variant to the protocol. | US main.tex Table 1 mean 0.857 (l.254); Table 3 mean 0.914 (l.385) |
| mad2-t1-patchcore-mvtecad, mad2-t1-patchcore-visa | medium. The protocol says both "ensemble backbone, 256x256" and "identical to EfficientAD Table 12". 92.7 / 79.7 equal EfficientAD's single-model PatchCore. EfficientAD's PatchCore-Ens gives 95.6 / 79.3. | Describe the column as single-model PatchCore (WRN-101, 224) per the value match. Mark the variant unclear. | MAD2 Table 1; MAD2 App. "PatchCore" (default ensembling); EfficientAD Table 12 |
| fastflow-t2-padim-img, fastflow-t2-padim-pix | medium. FastFlow's "PaDiM" cell (97.9, 97.5) mixes two variants. 97.9 is EfficientNet-B5 image AUROC; 97.5 is WR50-Rd550 pixel AUROC. PaDiM WR50 image AUROC is 95.3. | Label the variants in the protocol. Do not plot 97.9 as WR50 PaDiM. | FastFlow Table 2 (fastflow.tex l.452); PaDiM Table V (95.3 WR50, 97.9 EffNet-B5) and Table III (97.5) |
| fastflow-t2-spade-img | medium. "SPADE*" 96.2 is not SPADE's published image AUROC (85.5). It equals FastFlow Table 1's own WRN50+SPADE row. | Label it as a FastFlow-side SPADE variant. | FastFlow Table 1 (l.407); SPADE Table 1 Average 85.5 |
| draem-t1-us-img, draem-t2-us-pix, draem-t2-us-ap, cutpaste-t1-ustudent | medium. These numbers are not in the Uninformed Students CVPR paper, which reports only AU-PRO on MVTec AD. Neither DRAEM nor CutPaste cites the MVTec AD IJCV re-evaluation (VisA quotes 92.2 image AUROC for S-T from it). The two papers disagree (87.7 vs 92.5). | Values are as printed. The source is not stated (likely a re-run). Say so in the protocol. | US main.tex Tables 1 and 3 (PRO only); DRAEM and CutPaste bibliographies; CutPaste Sec. 4.1 text |
| draem-t1-padim-img, draem-t2-padim-pix, draem-t2-padim-ap | medium. These are DRAEM-side PaDiM runs, not PaDiM's published numbers. DRAEM gives per-class image AUROC and pixel AP, and PaDiM reports neither. | Our reading: re-run by DRAEM. PaDiM's own values are 95.3 / 97.5. | DRAEM tab:resSota, tab:resSotaLoc; PaDiM Tables III and V |
| padim-rdmvtec-drop-spade (8.8), padim-rdmvtec-drop-vae (12.2) | low. The text-stated drops do not match PaDiM's own tables. SPADE 96.5 - 87.2 = 9.3 (8.8 uses SPADE's own 96.0). VAE 74.4 - 62.1 = 12.3. | Keep the printed values. Add what they are relative to in the claim text. | PaDiM bare_conf.tex l.722; Tables III and VI |
| patchcore-hardware | low. "Nvidia Tesla V4" is printed, but no such GPU exists. | Quote it as printed or omit it. Set uncertain. | supplementary_arxiv.tex l.25 |
| map-subset (780) | low. The subset rule does not list the "image words". An independent re-implementation gives 786 (visual 819). The claim says 780 (813). | Record the exact regex next to the scan. | state.json arxiv_scan.subset_rule; /tmp/fca/scripts/mapcheck.py |

## What was checked

### Script pass (all 1,439 non-map claims)

- The script matched each value against the cited paper's LaTeX, pdftotext (layout and raw) and abs.html text. PatchCore also included code/ and README. The match allowed for thousands separators, percent vs fraction, dates in abs.html form and number words.
- Results: 1,288 exact hits, 43 numeric-token hits, 108 misses. Version check: every `arxiv` field matches the downloaded abs.html version.
- Misses by kind (all explained):
  - 96 our_reading computations. All recomputed below.
  - 7 DRAEM code_only values. The code is not in sources.
  - 3 FastFlow channel widths. These are flagged uncertain, and the loc says "not in paper".
  - 2 MVTec AD 2 sums. Recomputed below.
  - mad2-test-sets: formatting only (LaTeX macro).

### our_reading arithmetic (all recomputed, all correct)

- PatchCore banks (90 claims): train x 784, floored at 25/10/1% as in sampler.py; MB = n x 4096 / 1e6 (decimal MB). Per-class train counts match MVTec AD Table 1. Total 3,629 x 784 = 2,845,136.
- PatchCore shapes: 512x28x28, 1024x14x14, 784 x 1024, unfold 4,608 / 9,216.
- PaDiM: 550 + 550^2 = 303,050 floats; 1,212,200 B; x 3,136 = 3.80 GB. R18: 10,100 floats; 40,400 B; 0.127 GB. Channel width 1792 = 256+512+1024.
- SPADE gallery: 50 x 3,136 = 156,800.
- EfficientAD params from Tables 6-8: teacher-S 2,694,144; student-S 4,267,392; teacher-M 8,026,624; student-M 11,615,488; AE 1,096,320. Also checked: shapes 259/130/133/67/67/64, no-pad 56 and 58, and D = 1,572,864.
- MVTec AD 2 Table 4: train 2,528 + val 302 + test 5,174 = 8,004, which matches the stated total.
- MAD2 drop 92.4 - 52.6 = 39.8. MVTec AD totals: test good sums to 467 and defective to 1,258. VisA normals sum to 9,621.

### Semantic pass (965 table-derived claims)

- Scope: every claim whose loc cites a table or Fig. 4 and is not our_reading or code_only. Counts: benchmarks 225, spade-padim 97, patchcore 212, efficientad 264, flows-synthetic 116, branches-a 51.
- Method: parse each LaTeX table into rows. Read the rows next to the claim list and check row, column, variant and protocol (PDF text for MVTec AD and MemSeg).
- EfficientAD: Tables 1-5, 9, 12, 14, 16 and 17 checked in full. Every "MVTec AD" claim points at the Table 2 MAD column or Table 12, not at the 3-collection mean in Table 1. Latency is batch 1 and throughput is batch 16, both on RTX A6000 (App. E).
- PatchCore: Tables 1-6, S5 (all 91 low-shot cells) and S6 were checked, plus the README table. Tables 1-3 hold coreset 25/10/1%; Table 4 holds the 280/320 ensemble rows. Times include the backbone (Sec. 4.5).
- SPADE: all tables. SPADE's own pixel AUROC is 96.0; PaDiM prints SPADE as 96.5.
- PaDiM: Tables I-VIII. WR50-Rd550 and R18-Rd100 are in the right columns; image AUROC comes from Table V and pixel from Table III.
- Benchmarks: MVTec AD Tables 1 and 3 (all 75 per-class cells), Uninformed Students Tables 1 and 3, VisA Tables 1, 2, 3 and 6, and MVTec AD 2 Tables 1, 4, 7, 8, 9 and 11, plus the Fig. 4 values.
- FastFlow: Tables 1, 2 and 5-8. The ResNet18 main row uses the 3-3 subnet, WRN50 uses 3-1.
- CutPaste: Tables 1, 2, 3, 5 and 6. The 3-way, binary and ensemble columns are correct.
- DRAEM: Tables 1-4.
- Branch papers: all 19 papers' headline rows. PNI ensemble 99.63 matches the PDF; the LaTeX holds an unused v0 table with 99.62. ReConPatch Table 6 is 480x480.
- Dates: all 35 date claims match the abs.html version history.

### Map claims (recomputed with my own filter)

| claim | claimed | recomputed |
|---|---|---|
| map-unique | 1,165 | 1,165 |
| map-q-mvtec-abs / industrial / title-defect / visa | 406 / 189 / 927 / 153 | 406 / 189 / 927 / 153 |
| map-subset | 780 | 786 (rule underspecified) |
| map-fam-flow / memory / st | 19 / 80 / 52 | 19 / 80 / 52 |
| map-fam-vlm / fewshot | 202 / 103 | 204 / 102 (follows the subset difference) |
| map-year-* | sum 780 | sum 786; each year within 7 |

### Random sample

- 60 non-headline claims were drawn with random.seed(20261010) from the 263 claims whose text names no metric, size or count (PatchCore bank rows excluded).
- All 60 match their source and loc. One (draem-code-perlin-thr) is unverifiable; see below.

### Not checked semantically

- 282 text-stated constants are outside the table scope: benchmarks 76, spade-padim 43, patchcore 21, efficientad 59, flows-synthetic 62, branches-a 21. Examples: EfficientAD config values (listing lines 6-70), CutPaste App. A values, DRAEM Sec. 4.1 values.
- These passed the script pass (the value occurs in the cited paper). Only the 60 sampled claims also had their meaning and loc read.

### Unverifiable here

- 19 DRAEM code_only claims (draem-code-*, draem-params-*) cite github.com/VitjanZ/DRAEM @2dbf673. sources/draem/ has no code. No patch is proposed, because the commit is pinned and the values look plausible; the 3 params claims are already uncertain. Verify them before the page leans on them.

## Not problems, noted

- The EfficientAD Table 4 loc paraphrases the row label. The row is printed as "PDN".
- PatchCore states "a third of all classes" solved in Sec. 4.2 but "six of the 15" in Supp. B. The claim already records both.
- VisA appendix Table 7 prints the supervised mean AU-ROC as "7.80", a typo for 87.8. The claims cite main Table 2, which is correct.
