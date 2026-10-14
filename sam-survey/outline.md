# Segment Anything: the journey and the literature -- outline

Reader: basic deep learning only, explained from scratch. Already has explainers on ViT/CLIP, DINO and the
DETR family: link to those, do not re-teach them. Works on industrial visual inspection.
Pages: long page (no narration; cue ids and player-ready structure kept) + atlas. Figures are written once
as scenes (`SCENES.<id> = ctx => handle`) and mounted in both pages.

## Tiers (proposed; approve at checkpoint 2)

Spine (full read, one chapter or more each):
- SAM, 2304.02643v1 (2023-04-05)
- SAM 2, 2408.00714v2 (v1 2024-08-01, v2 2024-10-28; the v2 source is "SAM 2.1")
- SAM 3, 2511.16719v2 (v1 2025-11-20, v2 2026-03-28; v2 adds an "Object Multiplex" appendix)
- SAM 3D, 2511.16624v2 (v1 2025-11-20): short delta chapter (user's choice)

Prehistory (one delta each, abstract + method figure + one key fact):
- FCN 1411.4038v2, U-Net 1505.04597v1, Mask R-CNN 1703.06870v3, MaskFormer 2107.06278v2, Mask2Former 2112.01527v3
- Interactive segmentation: DIOS 1603.04042v1, RITM 2102.06583v1, SimpleClick 2210.11006v3 (FocalClick 2204.02574v2 on the tree)
- Video memory: STM 1904.00607v2, XMem 2207.07115v2 (Cutie 2310.12982v2 on the tree)
- Backbones SAM inherits: MAE 2111.06377v3, ViTDet 2203.16527v2, Hiera 2306.00989v1

Branches (abstract + one key table, one line each on the family tree):
- Efficient: MobileSAM 2306.14289v2, MobileSAMv2 2312.09579v1, EfficientSAM 2312.00863v1,
  EfficientViT-SAM 2402.05008v2, EdgeTAM 2501.07256v1, EfficientSAM3 2511.15833v1
- Quality: HQ-SAM 2306.01567v2
- Domain: MedSAM 2304.12306v3, MedSAM2 2504.03600v1, SAM-Adapter 2304.09148v3
- Composition: Grounding DINO 2303.05499v5, Grounded SAM 2401.14159v1, SEEM 2304.06718v4, Track Anything 2304.11968v2
- Video follow-ups: SAM2Long 2410.16268v3, SAMURAI 2411.11922v2
- Inspection and failure analyses: Industrial SAM 2307.12674v1, SAM in Defect Detection 2311.10245v1,
  Segment Any Anomaly 2305.10724v1, KairosAD 2505.24334v1, Boxes2Pixels 2604.11162v1,
  "Not Always Perfect" 2304.05750v3, "SAM Struggles in Concealed Scenes" 2304.06022v4

Map: arXiv scan 2026-10-09, title or abstract contains "segment anything": 1,447 unique records
(title query 436, abstract query 1,397). Counts by month + title-keyword domain split, labelled heuristic.
Pre-April-2023 hits are noise; 2026-09/10 is incomplete; papers that write only "SAM" are missed.

## Color key (one meaning each, whole page)
blue = image and image features; yellow = prompts (points, boxes, text, exemplars); green = predicted masks
and outputs; red = loss, error, the gradient; brown = frozen, discarded, background.

## Act I. Before SAM

### 1. ch-before -- Four ways to draw a mask
Intuition: segmentation went from "label every pixel" (FCN, U-Net) to "find each object, then mask it"
(Mask R-CNN) to "predict a set of masks, each with a class" (MaskFormer, Mask2Former), but every one of
them learns a fixed list of classes from one labelled dataset.
Visual (scene `before`, BUILT): one toy scene; tabs FCN / U-Net / Mask R-CNN / MaskFormer-Mask2Former redraw the stage as
that model's output and pipeline: a coarse-to-fine score map; the U with skip connections; box then mask;
N queries dotted with per-pixel embeddings into N masks. Toggle semantic / instance / panoptic views.
Toy, labelled. Delta cards with one sourced fact each.

### 2. ch-clicks -- A human in the loop
Intuition: interactive segmentation already had "prompts": clicks encoded as maps, the previous mask fed
back in, and training that simulates a user clicking on the errors (DIOS -> RITM -> SimpleClick). What it
lacked was scale and generality: a few datasets, one prompt type, one answer.
Visual (scene `clicks`): click inside/outside a toy object; the mask refines; the next simulated click
lands at the centre of the largest error region (the evaluation protocol; training samples the next point
uniformly from the error region, and the toy says so); an IoU-vs-clicks curve builds (NoC@90 explained). Toy.

## Act II. SAM (2023)

### 3. ch-promptable -- Any prompt, a valid mask
Intuition: SAM's task is "return a valid mask for any prompt". A point on a shirt is ambiguous (shirt,
person, the logo), so SAM returns three masks with predicted IoU scores instead of averaging them.
Visual (scene `promptable`, also the hero's core): click anywhere on a toy scene with nested objects; three
masks fan out (subpart, part, whole) with confidence bars. Button "one output only" shows the blurry
average a single-output model learns. Prediction prompt: what does one output do with an ambiguous click?

### 4. ch-sam-arch -- Heavy once, light many times
Intuition: an expensive image encoder runs once per image; a tiny prompt encoder and mask decoder run per
prompt in tens of milliseconds, so interaction feels real time.
Visual (scene `samflow`): ArchFlow forward stepper: 1024x1024 image -> ViT-H/16 (MAE init) -> 64x64x256
embedding; points/boxes -> positional encoding + learned type embeddings; mask prompt -> convs; two-way
transformer (2 layers) -> upscaled embedding -> dot with output tokens -> 3 masks + IoU scores. Backward:
focal + dice (20:1) on the best of 3 masks only for a single prompt; with more than one prompt a 4th
output token returns one mask and there is no min; MSE on IoU predictions; everything trained (encoder
from MAE, layer-wise LR decay). Decoder: 2 two-way layers, then one final token-to-image attention.
The paper prints no loss equation: the page's loss equation is labelled "our transcription of App. A".
ViT-H width/depth/heads come from the ViT/MAE papers (or marked uncertain).
Shapes computed from the paper's config. Equations: the loss with min over masks.

### Interlude ch-training -- One training step: eleven simulated rounds (not narrated)
Stepper: sample mask -> first prompt (point or box) -> predict 3 masks -> next point sampled uniformly
from the error region -> previous logits fed back as a mask prompt -> one mask (4th token) -> ... 11 rounds
(8 sampled points, 2 rounds with no new point) -> loss. Note what SAM 2 changes (7 correction clicks, no
empty rounds, l1 IoU loss on all masks).
Readouts computed from the paper's recipe.

### 5. ch-engine -- The data engine and SA-1B
Claim card + interactive. Three stages (assisted-manual, semi-automatic, fully automatic), with masks,
images and seconds per mask per stage. Visual (scene `engine`): the fully automatic stage live on a toy
image: a 32x32 point grid (slider for grid density), each point -> 3 masks, filters (predicted IoU,
stability, the >=95%-of-image filter, NMS, the 20 zoomed crops, the 100-pixel cleanup) peel away the bad
ones; a stage-tab flywheel shows the model and data growing together (masks per image 20 -> 44 -> 72;
seconds per mask only for stages 1-2; the SA-1B generator is a special 3-layer-decoder SAM trained on
stage 1+2 data; the released SAM is trained on SA-1B).
Mask-quality check (sampled masks re-corrected by annotators) as the claim card.

### 6. ch-zeroshot -- What prompting buys
Claim card: 23-dataset single-point benchmark vs RITM (automatic mIoU and human ratings), then the
compositions (edges, proposals, instance segmentation from ViTDet boxes, text-to-mask via CLIP).
Visual: sourced bar chart per dataset (SAM minus RITM, printed on Fig. 9a bars), toggle oracle (best of
3; values read from the figure's vector file, labelled as such). Human study covers 7 of 23 datasets. At
5-9 points SAM is slightly below RITM/SimpleClick (the paper says "on par"). Keep LVIS v0.5 and v1 apart.
Limitations from Sec. 8: fine structures, small disconnected parts, boundaries less crisp than zoom-in methods.

## Act III. Time, concepts, depth

### 7. ch-sam2 -- Video is images with memory
Intuition: process frames as a stream; each frame's features attend to a memory bank of recent frames and
prompted frames, plus object pointers; a click on any frame corrects the whole track. Occlusion head.
Visual (scene `sam2`): a frame strip; the object moves, gets occluded, returns; the memory bank (FIFO of N
recent + prompted frames; no number shown for the prompted-frame count, which the paper does not give)
fills; a correction click on a later frame re-propagates. ArchFlow block with Hiera encoder, memory
attention, memory encoder, plus the stride-4/8 features that skip memory attention to the decoder, the
object pointer (mask token split into 4 x 64) into the memory bank, and the occlusion embedding; temporal
encoding only on the recent memories. "6x faster" is images only. Image = video with empty memory. Claim card: SA-V data engine phases
(annotation time per frame by phase) and SA-V scale.

### 8. ch-sam3 -- Every object of this kind
Intuition: SAM 3 changes the prompt from "this object" to "this concept" (a short noun phrase or image
exemplars) and returns every instance. A DETR-style detector (queries, link to the DETR explainer) shares a
Perception Encoder backbone with a SAM 2 style tracker; a presence token decides "is the concept here at
all" separately from "where is each instance".
Visual (scene `sam3`): type or pick a concept on a toy scene; per-query scores x global presence score;
hard negatives (similar-looking things) get suppressed; exemplar boxes add positives/negatives. The hard-negative toggle changes the "is it here" gate, not mask
quality (IL_MCC up, pmF1 flat); the presence head trades a little pmF1 for IL_MCC (Table 9a); show both.
Ambiguity head (2 experts) links back to SAM's three masks. Tracker rules as a mini-stepper
(confirmation delay, duplicate removal, periodic re-prompting). Object Multiplex (v2 appendix) as a card.
ArchFlow block. Claim card: SA-Co data engine with AI verifiers; the cgF1 metric; label the paper's
internal inconsistencies (LVIS 48.8 vs 48.5, ~850M vs 848M params).

### 9. ch-sam3d -- From a mask to an object
Delta: SAM 3D takes an image and a SAM mask and generates the object's 3D shape, texture and pose
(geometry model for coarse voxels + layout, then texture/refinement), trained like an LLM: synthetic
pretraining, then real-world alignment from a human-and-model-in-the-loop data engine with preference
optimization. Visual (scene `sam3d`, pe3): a mask lifts into a coarse voxel shape, then is placed with
rotation, translation and scale in the scene. Toy, labelled.

## Act IV. The family

### 10. ch-efficient -- SAM on a budget
Intuition: almost all of SAM's cost is the image encoder, so the efficient variants replace it:
distil a small encoder to match SAM's embedding (MobileSAM), pretrain the small encoder to reconstruct SAM
features (EfficientSAM), redesign the encoder (EfficientViT-SAM), and do the same for video memory
(EdgeTAM) and concepts (EfficientSAM3). HQ-SAM goes the other way: a tiny add-on for fine structures.
Visual: sourced chart of encoder parameters and reported speed per paper (each paper's own hardware,
labelled; no cross-paper ranking implied) + a distillation toy (student embedding pulled to teacher).

### 11. ch-family -- Compose, adapt, track
Composition (Grounding DINO boxes -> SAM = Grounded SAM; SEEM), adaptation (MedSAM fine-tuning, adapters),
video before SAM 2 (Track Anything) and after (SAM2Long, SAMURAI). Visual: the family tree / timeline SVG,
2014-2026, spine highlighted, branches one line each.

### 12. ch-map -- The explosion
Computed from the scan: counts per month since April 2023, title-keyword domain split (medical, remote
sensing, video, 3D, efficient, industrial, ...), labelled heuristic; every count a data-claim with the query.

## Act V. Your use

### 13. ch-inspection -- SAM for inspection: where it helps, where it fails
Claim cards from the inspection papers: label acceleration (box prompts, Boxes2Pixels noisy-mask training),
zero-shot anomaly segmentation needs prompt regularization (SAA), low-contrast and concealed defects
and thin structures fail ("Not Always Perfect", concealed scenes, SAM-Adapter), edge deployment (KairosAD,
efficient encoders). Practical recipe for a visual inspection platform, labelled "our reading".

## Closing
ch-timeline (SVG, sourced dates), ch-summary, ch-deeper (paper, section, equation per item).

## Atlas (8 stages, scenes shared with the long page)
overview (hero: click -> three masks), before, clicks, promptable, engine, sam2, sam3, inspection.
