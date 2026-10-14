# Understanding: Segment Anything, the journey (2026-10-09)

Per-paper notes hold the detail, the equations verbatim and every location: `notes/papers/prehistory.md`,
`sam-model.md`, `sam-data.md`, `sam2.md`, `sam3.md`. Claims: `notes/claims/*.json` (927 at this point; the
ones marked uncertain are never shown as fact). Branch, interactive, video-memory, SAM 3D and inspection
papers are extracted after checkpoint 2.

## The story in one paragraph
Before 2023, a segmentation model learned a fixed class list from one labelled dataset (FCN 21 VOC classes,
Mask R-CNN 80 COCO classes, Mask2Former 133 / 80 / 150 and still retrained per task and dataset). Interactive
segmentation had already found the other half: let a person point, encode the clicks, feed the previous mask
back, and train by simulating a user who clicks on the errors (RITM). SAM (2304.02643v1) fuses the two into a
task, "return a valid mask for any prompt", that is both the pre-training objective and the interface, and
makes it trainable at scale with a data engine that ends fully automatic (SA-1B: 11M images, 1.1B masks,
99.1% automatic). SAM 2 (2408.00714v2) treats an image as a one-frame video and adds a streaming memory, so a
click on one frame segments the object in every frame. SAM 3 (2511.16719v2) changes the prompt from "this
object" to "this concept" (noun phrase or exemplar boxes) and returns every instance, by putting a DETR-style
detector with a presence token next to a SAM 2 tracker on one shared Perception Encoder backbone. SAM 3D
(2511.16624v2) takes an image and a mask and generates the object's 3D shape, texture and pose.

## Lens per paper
- FCN, U-Net, Mask R-CNN, MaskFormer, Mask2Former: architecture lens, light (delta cards).
- SAM model (ch-promptable, ch-sam-arch, ch-training): architecture lens. Shapes are computable from App. A
  except the ViT-H width/depth/heads (standard ViT-H values; source them from the ViT/MAE/ViTDet papers or mark
  uncertain) and the final logit upsampling (code-only, not shown).
- SAM data and zero-shot (ch-engine, ch-zeroshot): dataset/analysis lens, claim cards.
- SAM 2: architecture lens leads (memory attention, memory bank, object pointers, occlusion head); the SA-V data
  engine is a claim card.
- SAM 3: both. Architecture lens leads for the detector, presence head, tracker, training stages; the dataset
  lens leads for the SA-Co data engine, the cgF1 metric and the human comparison.

## Key mechanisms, plainly
1. Ambiguity (SAM): a single-output model averages valid masks for an ambiguous click. SAM predicts 3 masks
   and an IoU score each; the loss uses only the lowest-loss mask, so each output slot specialises (in effect
   subpart, part, whole). With more than one prompt ambiguity is "much rarer" (App. A), and a 4th token returns one mask.
2. Cost split (SAM): ViT-H/16 encoder once per image (1024x1024 -> 256x64x64); prompt encoder + two-way
   transformer decoder per prompt, about 50 ms in a browser on CPU.
3. Training as simulated interaction (SAM): 11 rounds per mask: a first point or box, then 8 points sampled
   from the error region with previous logits fed back, plus 2 rounds with no new point (sam-model.md, Sec. 5).
4. Data engine (SAM): assisted-manual -> semi-automatic -> fully automatic; masks per image 20 -> 44 -> 72;
   fully automatic = 32x32 point grid, a set of masks per point (3 output tokens, our reading), filtered by
   predicted IoU, stability and NMS; zoomed crops add masks for small objects.
5. Memory (SAM 2): frame embedding cross-attends to a FIFO of 6 recent memories (with temporal position
   encoding) plus prompted-frame memories and object pointers; occlusion head says when the object is absent.
6. Presence (SAM 3): a global presence token predicts "is the concept in the image"; each query's score is
   conditioned on it, so recognition and localisation are learned separately; only the presence head learns
   absence (query scores get no gradient when the concept is absent).

## Known conflicts in the sources (label, do not reconcile)
- SAM: Fig. 18f OVIS legend swaps SAM and single-output ratings; Table 4 vs text on large objects.
- SAM 2: 61.4 (prose) vs 61.9 (Tables 5, 15) mIoU; "3.4x / 1.5x faster" not derivable from Table 15; masklet
  phase counts exceed the 190.9K manual total; 14 s vs 13.8 s average duration.
- SAM 3: LVIS mask AP 48.8 (Sec. 1) vs 48.5 (Table 1); SA-Co/EXT 15 vs eighteen datasets; ~850M vs 848M params;
  207K vs 270K benchmark concepts (paper vs README).
- MaskFormer: ADE20K-Full 874 (text) vs 847 (Table 2 header); the page uses 847.
