# Fact-check, page pass 1

Scope: parts/body-top.html, ch-compare, ch-map, ch-infonce, ch-moco, ch-simclr, ch-step, and kits 05, 10, 20, 25, 26, 30, 40, 41, 50-53.

Sources: the project folder has no sources/ directory. I fetched the arXiv versions named on the page again (e-print LaTeX and PDF) into /tmp/fc1/src/: cpc, wu2018, poole, tschannen, cpcv2, cmc, arora, moco, mocov2, mocov3, simclr, simclrv2, infomin, facenet, byol, wangisola, supcon, clip. I also fetched the Hadsell 2006 PDF from the URL on the page and the Sohn 2016 PDF from papers.nips.cc. I did not execute any downloaded file.

## Counts

- Static data-claim spans in scope: 330. Each span text matches its claim value in notes/claims.json (0 mismatches). The 3 dynamic spans (tree dates, scan counts, Fig. 5 cells) come from data that I checked separately.
- Tree dates: I checked all 41 arXiv first-version dates in kit/26-tree.js against the arXiv API `published` field. All 41 match.
- Scan counts: KIT.MAPSCAN matches notes/scan/arxiv-counts-2026-10-13.json row by row. The shares in the prose (27%, 17%, 18%) and the trend statements in the prose are correct.
- Equations checked against the source: DrLIM Eq. 4, CPC Eq. 3 and Eq. 4, the CPC bound display and App. A Eq. 8, Poole Eq. 10 (Sec. 2.3), MoCo Eq. 1 and Eq. 2, SimCLR Eq. 1, and Alg. 1. All match, and the eqno lines are correct.
- Recomputed arithmetic: the lineup example (0.40, 0.90, 0.29, 1.24, 0.48, 0.99, 0.008, 0.71, 0.10, 5.55, 4.85), MoCo momentum (693, 23%, 0.7%, 0.0008, 0.49, 0.40), and NT-Xent (0.79, 0.23, 0.59, 0.53). All are correct.
- Computed figures:
  - Hero: re-ran build/sim/hero2.js. The pair gap reaches 0 and the spread reaches 0.99, so "pairs meet and spread evenly" holds.
  - Pullpush: the gradient is correct for DrLIM Eq. 4, and the readout matches notes/sim-pullpush.txt.
  - Lineup: exact softmax.
  - ntxent: exact NT-Xent, and it matches notes/sim-simclr.txt.
  - Queue toy: runs Eq. 2 exactly.
  - Step: shapes are exact; the losses are toy values, and the caption says so.
  - The badges are honest, with the exception of the hero aria-label below.
- Problems: 8 in total: 1 WRONG, 5 MISLEADING, 1 OVERSTATED, 1 UNSOURCED.

## Problems

1. parts/ch-infonce.html, Tschannen claim card, Test 1: "An invertible encoder trained adversarially ends up worse than raw pixels while it still maximizes the bound." | WRONG | tschannen main.tex l.247-251: the adversarial run is trained "without using a MI estimator", so it computes no bound. What stays maximal is the true MI ("despite still achieving globally maximal MI"). | Fix: "An invertible encoder trained adversarially (no MI estimator) ends up worse than raw pixels, although its true mutual information stays at the maximum."

2. parts/ch-moco.html, MoCo v2 paragraph: "A two-layer MLP head ... replaced the single layer; alone it added 5.6 points (60.6 to 66.2)." | MISLEADING | mocov2 main.tex l.173-181: "Using the default tau = 0.07, pre-training with the MLP head improves from 60.6% to 62.9%; switching to the optimal value for MLP (0.2), the accuracy increases to 66.2%." Row (a), 66.2, already includes tau = 0.2. The claim mocov2-derived-mlp-gain ("MLP head alone") carries the same error. | Fix: "replaced the single layer: at the old temperature 0.07 it added 2.3 points (60.6 to 62.9), and with the temperature raised to 0.2 it reached 66.2 (+5.6)." Then drop "The temperature rose ..." later in the paragraph, or keep it as a back-reference.

3. parts/ch-step.html, stage 8: "each negative pulled into the update with a weight equal to its softmax probability" | MISLEADING (minor) | simclr main.tex l.375 (Table 2): the NT-Xent gradient weights each v- by exp(u^T v-/tau)/Z(u)/tau, so the weight is the probability divided by tau. | Fix: "with a weight proportional to its softmax probability (the probability divided by tau)".

4. parts/ch-simclr.html, viz-augment figcaption: '"Color" is SimCLR's color distortion (brightness, contrast, saturation and hue jitter)' | MISLEADING (minor) | simclr main.tex l.302 and App. A l.605: color distortion "is composed by color jittering and color dropping". kit/51-augment.js sets grey:false, so the toy draws jitter only. | Fix: '"Color" is drawn as jitter only (brightness, contrast, saturation, hue); SimCLR's color distortion also turns the view grey with probability 0.2.'

5. parts/body-top.html, hero canvas aria-label: "Then red arrows push each point away from the points of other images." | MISLEADING (minor) | kit/10-hero.js draws the push arrows from image 0's anchor only (`const [ax, ay] = P(0)`). | Fix: "Then red arrows push one point away from the points of all other images."

6. parts/ch-infonce.html, Tschannen claim card, Test 2: "The MLP critic gives the tightest bound but barely beats raw pixels; the simple bilinear critic is the loosest and gives the best features." | MISLEADING (minor) | tschannen main.tex l.289: the tightness ranking is for the NWJ (MINE) estimate. The InfoNCE value "is close to the theoretically achievable maximum value for all critics". | Fix: "Measured with the NWJ bound, the MLP critic is tightest but barely beats raw pixels; the bilinear critic is loosest and gives the best features."

7. parts/ch-compare.html, last section: "Later work showed that, when the score is a dot product of two embeddings, it is the same objective as InfoNCE" (similar wording in ch-infonce.html l.112: "with a dot-product critic, it is the N-pair metric-learning loss") | OVERSTATED (minor) | tschannen main.tex l.329-340: the equivalence holds for a symmetric separable critic phi(x)^T phi(y) with one encoder shared across views, "up to constants and change of sign". With different encoders per view, you get only "asymmetric variants". | Fix: "when both inputs go through the same encoder and the score is the dot product of the two embeddings, it is the same objective as InfoNCE, up to constants".

8. parts/ch-infonce.html, viz-cpcflow stage 1: "From each ImageNet image, resized to 300x300, CPC takes a random 256x256 crop" | UNSOURCED (minor) | cpc experiments.tex l.139: "The 256x256 images are randomly cropped from a 300x300 image". The paper does not say how it gets the 300x300 image. | Fix: "From a 300x300 version of each ImageNet image, CPC takes a random 256x256 crop".

## Checked and sound (no action)

- DrLIM attribution and the name "contrastive loss". Sohn N-pair: the batch of N pairs from N classes, the reuse of other pairs as negatives, and the tests on Online Products and LFW faces.
- Not verified against a primary source (no PDF fetched): Chopra 2005 (title known only from Hadsell ref. [5]), Gutmann 2010, and the FaceNet one-line summary. Only the venues and one-line summaries are involved.
- MoCo v2 "took two things from SimCLR, added a cosine schedule": this matches the paper's own framing (abstract: two SimCLR design improvements; cosine added "for fair comparisons", l.190). No change.
- CPC: the "categorical cross-entropy" and "becomes tighter as N becomes larger" quotes. Sec. 3.2 image setup: 7x7 grid, 64x64 patches with 32-px overlap, 60x60 subcrop, ResNet-v2-101 without BN, third-block 1024-d features, PixelCNN-style context, up to five rows, Adam 2e-4, 32 GPUs x 16, 48.7. The negatives source is unstated for images, and the page says so. The CPC v2 quote is exact.
- Tschannen: the quotes "cannot be attributed to the properties of MI alone" and "can result in worse representations", and the numbers 128, 20 runs, 85%, 94%, 30 coupling layers, 40k, t = 2/4, 7k, 4.85. Sec. 4 is the metric-learning section.
- Wu 2018: tau 0.07, m = 4096, 128-d, 600 MB, batch 256, 54.0 (conv5, linear SVM). Arora: tau is the class-collision probability, Thm 4.1.
- MoCo: the Eq. 2 quote "Only the parameters theta_q ...", the 1024 max batch, the "oscillates and fails to converge" quote, the momentum table, Fig. 3 and its caption (K-1 for end-to-end; the improved memory bank uses InfoNCE), shuffle BN (>99.9%), Sec. 3.3 augmentation, and lr / wd / 8 GPUs / 53 h.
- MoCo v2: 2048 hidden, blur added, color distortion "diminishing gains", cos +0.2, 71.1 at 800 ep, Table 3 memory (93.0G marked as an estimate).
- MoCo v3: no queue at 4096, m 0.99, prediction head, 2*tau scale, m = 0 gives 74.3 "analogous to SimCLR", +2.2, batch 2048 to 6144 gives 72.6 to 69.7, spikes first in the patch projection, +1.7 at lr 1.5e-4, 76.5 / 76.7.
- SimCLR: the default setting (no default tau stated; no head hidden width stated), global BN, Fig. 5 protocol and the "hurts" note, Table 1 (supervised 90 ep), Table 3 (2048-d, random-guess column), Sec. 4.2 (+3%, >10%, >10%), Table 5, Table B.1 (left of slash = linear scaling), App. B.3, 16382 at 8192, LARS / lr / warmup / TPU.
- InfoMin: 0.2 sweet spot, 73.0 with the PIRL jigsaw. SimCLRv2: R152 3x+SK, 3-layer head, 64K memory, 800 ep, 79.8, 73.9 / 77.5 distilled R50.
- ch-map "our reading" statements follow from the counts. The tree summaries and "builds on" edges are consistent with the paper notes and the sources I checked.
