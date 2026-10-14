# wang2019fgfi -- Distilling Object Detectors with Fine-grained Feature Imitation (BRANCH)

## Identity
- arXiv 1906.03609, version read: v1 (only version), submitted 2019-06-09 (abs.html).
- Authors: Tao Wang, Li Yuan, Xiaopeng Zhang, Jiashi Feng. Venue: CVPR 2019 (abs.html comments "accepted at
  CVPR2019").
- Repo: https://github.com/twangnh/Distilling-Object-Detectors (abstract). README and `trainval_net_sup.py`
  fetched 2026-10-09 into `sources/wang2019fgfi/README.md` and `sources/wang2019fgfi/code/` (read only). Repo
  covers the VGG11 Faster R-CNN student; README says its numbers "are different from the paper".

## Key idea (two sentences)
Plain logit KD gives a detector little (few classes, no localization signal), and imitating the whole feature
map (FitNets hints) hurts, because background locations dominate and are noisy. So the student imitates the
teacher's last anchor-level feature map only at "near object anchor locations", chosen per object by an IoU
threshold relative to that object's best anchor, through a learned adaptation layer.

## Imitation region (Sec. 3.1)
For each ground-truth box, compute IoU with all anchors -> W x H x K IoU map m (K = anchors per location).
M = max(m); filter threshold F = psi * M; keep locations with IoU > F, OR over the K anchors -> W x H mask; OR
over all gt boxes -> imitation mask I. psi = 0 keeps every location (= full feature imitation); psi = 1 keeps
none. "In all experiments, a constant $ \psi =0.5$ is used." A fixed F is rejected because "object size usually
varies in a large range".

## Equations (TeX verbatim, `src/egpaper_final.tex`; numbers from PDF Sec. 3.2)
Eq. 1, per location:
```
l =
\sum_{c=1}^C(f_{\mathrm{adap}}(s)_{ijc}-t_{ijc})^2,
```
Eq. 2, imitation loss:
```
&L_{imitation} = \frac{1}{2N_{p}}\sum_{i=1}^W\sum_{j=1}^H\sum_{c=1}^CI_{ij}(f_{\mathrm{adap}}(s)_{ijc}-t_{ijc})^2, \\&\text{ where } N_{p} = \sum_{i=1}^W\sum_{j=1}^HI_{ij}.
```
Eq. 3, total loss:
```
L = L_{gt} + \lambda L_{imitation},
```
Paper: "$I$ is the imitation mask, $N_{p}$ is the number of positive points in the mask, $f_{\mathrm{adap}}(\cdot)$
is the adaptation function ... $L_{gt}$ is the detection training loss and $\lambda$ is imitation loss weight
balancing factor."
Gloss: s = student guided feature map; t = teacher feature map (same W x H after adaptation); f_adap = "full
convolution adaptation layer"; squared error summed over channels at masked locations, normalized by twice the
mask count.
- lambda value: NOT stated in the paper. Official code default `--ilw` = 0.01 (`code/trainval_net_sup.py.txt`
  lines 121-123); imitation off for the first 400 steps of epoch 1 (`--ws` default 400, lines 117-119 and
  387-392). Both `code_only`. Code normalizer `norms = mask_batch.sum() * 2` (line 382) matches Eq. 2.
- Teacher fixed: not stated in the paper. Code: `fasterRCNN_sup_vgg16.eval()` and `param.requires_grad = False`
  for all teacher params (lines 276-282). `code_only`.

## Results (all AP numbers absolute points; "-I" = with imitation)
Table 1 (KITTI val, toy ShuffleNet-based one-stage detector; teacher 1x, students = channels halved once or
twice; columns Flops/G, Params/M, car/pedestrian/cyclist Easy/Moderate/Hard, mAP):
| Model | Flops/G | Params/M | mAP |
|---|---|---|---|
| 1x (teacher) | 5.1 | 1.6 | 62.63 |
| 0.5x | 1.5 | 0.53 | 57.98 |
| 0.5x-I | 1.5 | 0.53 | 60.46 (+2.5) |
| 0.25x | 0.67 | 0.21 | 45.96 |
| 0.25x-I | 0.67 | 0.21 | 52.63 (+6.7) |
| 0.25x-F (full feature imitation = hints) | 0.67 | 0.21 | -8.9 (delta) |
| 0.25x-G (scaled gt boxes as region) | 0.67 | 0.21 | +2.0 (delta) |
| 0.25x-D (vanilla KD on outputs only) | 0.67 | 0.21 | +0.9 (delta) |
| 0.25x-ID (imitation + KD) | 0.67 | 0.21 | +5.2 (delta) |
Prose (Sec. 4.2) says the 0.25x model "is boosted by 6.6 mAP" while the table delta reads +6.7. Both as printed.
The PDF text refers to this table as "Table 3.2" (label placed before caption in TeX); the caption is Table 1.

Table 2 (Pascal VOC07, Faster R-CNN, halved student, mAP): res101 74.4; res101h 67.4; res101h-I 71.2 (+3.8).
Table 3 (Pascal VOC07, Faster R-CNN, shallow student, mAP): VGG16 70.4; VGG11 59.6; VGG11-I 67.6 (+8.0).
res101 74.4; res50 69.1; res50-I 72.0 (+2.9). "nearly recovers 74% of the performance drop" (VGG11 case, relative).
Table 4 (COCO, Faster R-CNN ResNet-101 teacher, halved student):
| Model | AP@0.5 | AP | APs | APm | APl | AR | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|
| res101 | 54.6 | 34.4 | 14.3 | 39.1 | 51.9 | 45.9 | 23.0 | 52.2 | 66.4 |
| res101h | 48.4 | 28.8 | 11.8 | 32.0 | 44.9 | 41.5 | 19.8 | 45.9 | 62.3 |
| res101h-I | 51.2 | 31.6 | 13.2 | 35.9 | 47.5 | 44.0 | 22.4 | 50.3 | 64.5 |
Table 5 (COCO, multi-layer imitation, ResNet-50 FPN Faster R-CNN teacher, halved student; mask computed per
FPN level with that level's anchors):
| Model | AP@0.5 | AP | APs | APm | APl | AR | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|
| res50 | 59.0 | 36.9 | 21.5 | 39.8 | 48.3 | 50.5 | 31.4 | 53.9 | 63.6 |
| res50h | 52.6 | 31.2 | 18.5 | 32.0 | 42.4 | 46.3 | 27.7 | 47.5 | 60.6 |
| res50h-I | 55.8 | 34.8 | 21.0 | 34.9 | 45.5 | 49.1 | 30.5 | 52.6 | 63.5 |
(delta row: +3.2 AP@0.5, +3.6 AP.) COCO split, epochs, lr and GPU are not stated in the paper.

Supporting analysis: psi sweep (Fig. 6a, Sec. 4.4.4): 0.5 best; psi = 0 (full imitation) is "even much lower
than the ones trained with only ground truth supervision". Background variance (Sec. 4.4.5): full-imitation loss
"more than ten times" the masked loss.

## Outline feedback
- Best sourced row for the chapter: Table 5, the ResNet-50-FPN Faster R-CNN (the outline's running-example
  family): halved student 31.2 -> 34.8 COCO AP, teacher 36.9.
- Table 1 rows -D (+0.9) and -F (-8.9) are the sourced basis for "logits are not enough" and "whole-map hints
  hurt a detector".
- lambda is code-only (0.01); viz-distflow should say so if it shows the weight.
- The student here is the same detector family with halved channels or a shallower backbone, not a different
  architecture.
