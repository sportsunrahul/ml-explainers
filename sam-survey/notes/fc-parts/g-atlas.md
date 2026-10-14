# (e) Atlas fact check: atlas/chapters.html (+ hero.html, chips.html, dialogs.html)

Method: every number and factual sentence was matched to a claim in notes/claims.json and then to the
source line. Paths below are relative to sources/.

## WRONG

1. Panel 8 (SAM for inspection): "even with the best of its masks picked by hand, SAM ViT-H has a mean
   absolute error of 0.265". The masks were not picked by hand. The paper runs SAM in automatic
   (everything) mode to get N masks and selects the mask with the highest IoU against the ground truth
   (notperfect/latex/main.tex:665). Correct: "even when the best of its automatic masks is chosen by
   comparing each with the ground truth (an oracle upper bound)". The long page states this correctly
   (explainer.src.html:3151). The preceding "Clicked once, SAM tends to return the object around the
   defect" also does not match this protocol: the 0.265 row uses everything mode, not one click.

2. Panel 4 (Any prompt, a valid mask): "With two clicks or a box, the ambiguity is much rarer, and a
   fourth output returns one mask." The paper ties the fourth token to "more than one prompt"; the
   fourth mask "is never returned for a single prompt" (sam/latex/segany.tex:654). In SAM's training a
   box is one prompt: the first prompt is "either a foreground point or bounding box"
   (segany.tex:658). The paper does not say a box alone uses the fourth token. The long page avoids
   this: its Box mode is labelled toy (explainer.src.html:1414). Correct: "With two or more prompts
   (for example two clicks), ambiguity is much rarer, and a fourth output returns one mask."

3. Panel 3 (A human in the loop): "RITM (2021) added the two ideas SAM reuses: it feeds the previous
   mask back in, and it trains by simulating a user who clicks on the errors." RITM did not add either
   idea. Its title is "Reviving Iterative Training with Mask Guidance", and it says "We revive an
   iterative training procedure, and make a network aware of the mask from a previous step
   [Mahadevan et al. 2018]" (ritm/latex/main.tex:98). Both the iterative click simulation and the
   previous-mask input are from Mahadevan et al. (ITIS, 2018) (ritm/latex/main.tex:184, 191).
   Correct: "RITM (2021) revived and refined two ideas from ITIS (2018) that SAM reuses: ...". SAM
   itself cites RITM and Forte et al. for the simulated setup (segany.tex:658).

## MISLEADING / over-claimed

4. Panel 2, Mask2Former paragraph: "A fixed set of queries (100 by default) ... 57.8 PQ on COCO
   panoptic." The paragraph covers MaskFormer and Mask2Former, but 57.8 PQ is Mask2Former with a
   Swin-L backbone, 200 queries and 100 epochs (mask2former/latex/sections/expts.tex:25; abs.tex:11).
   MaskFormer Swin-L gets 52.7 PQ in the same table (expts.tex:23). Name the model and backbone, and
   do not set the number next to "100 by default".

5. Panel 2 takeaway: "Every model before SAM learned a fixed list of classes from one labelled
   dataset, and was retrained for each new task." The next panel contradicts it: interactive
   segmentation before SAM needs no class list (RITM 2021, SimpleClick 2022). The long page limits the
   claim to "segmentation models" of that family (explainer.src.html:1057, 3261). Correct: "Every
   model in this panel ...".

6. Panel 6 (SAM 2): "a memory bank of the 6 most recent frames and of every frame you prompted". The
   paper keeps "a FIFO queue of up to M prompted frames" (sam2/latex/sam2.1_arxiv.tex:319). "Every"
   over-claims. (Note for the page fork: explainer.src.html:2239 says "The paper does not state how
   many prompted memories are kept"; the paper names the bound M but gives no value.)

7. Panel 7 (SAM 3): "Splitting 'is it here?' from 'where is each one?' is what lets SAM 3 say 'no
   yellow mug'". The paper credits both the presence head and training with hard negatives for
   rejecting absent concepts: the presence head is "especially effective when training with
   challenging negative phrases" (sam3/latex/main.tex:353), and hard negatives separately improve
   IL_MCC (main.tex:983). "Is what lets" is our reading; soften to "helps SAM 3 say".

8. Panel 8: "For thin defects, HQ-SAM's fine-detail token helps." No source in the download tests
   HQ-SAM on defects. HQ-SAM evaluates on general fine-structure sets; the only inspection paper that
   mentions HQ-SAM (industrialsam/latex/paper.tex:132) only cites it in related work. Label as our
   reading, or cite Boxes2Pixels' finding that SAM misses thin cracks (boxes2pixels/latex/sec/
   3_method.tex:7) as the motivation only.

9. Panel 8: "Clicked once, SAM tends to return the object around the defect." The source for this
   behaviour (Concealed, concealed/latex/main.tex:297, "SAM's behaviour appears to segment the main
   part of the object, such as the screw ... and bottle") used the online demo with no stated prompt
   mode (main.tex:293). "Clicked once" is not stated. Minor.

10. Panel 8: "the industrial defect set CDS2K ... 0.089 for a dedicated model". DGNet is a camouflaged
    (concealed) object detector, the best of the listed SOTA rows on CDS2K (notperfect/latex/
    main.tex:619), not a defect-specific model. CDS2K is a "concealed defect segmentation dataset"
    (main.tex:459). Minor; "the best dedicated model in the table (DGNet)" is safer.

11. Panel 3: "RITM needs 5.34 on DAVIS, SimpleClick 4.78." Both are correct and comparable (both
    trained on COCO+LVIS, DAVIS NoC@90). The RITM row is H32 IT-M (ritm/latex/main.tex:512, DAVIS
    NoC@90 column) and SimpleClick is ViT-H (simpleclick/latex/main.tex:266). The variants are not
    named; the ViT-H vs HRNet-32 size gap is large. Minor.

12. Panel 6: "SAM 2 is about 6 times faster than SAM". The 6x compares SAM 2 Hiera-B+ with SAM ViT-H
    (sam2/latex/sam2.1_arxiv.tex:1442; abstract). Hiera-L is 3.4x. Variant not named. Minor.

## Checked and correct

- Panel 1: three masks with predicted scores (segany.tex:652); prompts click / box / rough mask.
- Panel 2: FCN 59.4 -> 62.7 mean IU, FCN-32s vs FCN-8s on a VOC2011 val subset (fcn/latex/paper.tex:598-600;
  59.4 uses extra training data, paper.tex:483). U-Net 30 EM training images (unet/latex/paper459.tex:163)
  and 2 output classes (Fig. 1). Mask R-CNN 28x28 mask, 80 COCO classes. Queries 100 by default.
  Class table 21 / 80 / 133 / 150 matches claims fcn-voc-classes, maskrcnn-mask-classes,
  m2f-coco-pan-classes, m2f-ade-classes. Years 2014 / 2015 / 2017 / 2021 match the timeline dates.
- Panel 3: NoC@90 definition and the click rule "centre of the largest error region" (ritm/latex/main.tex:320).
- Panel 4: loss on all three masks, backprop from the lowest (segany.tex:652); ~50 ms in a browser on CPU
  (claim sam-decoder-runtime); ViT-H encoder runs once per image.
- Panel 5: 1.1B masks, 11M images, 99.1% automatic, 20 -> 44 -> 72 masks per image, 32x32 grid,
  confident / stable / NMS (segany.tex:208), 95% whole-image filter (segany.tex:675), holes and
  components < 100 px (segany.tex:677). "Up to three masks" per point is derived from the 3 output
  tokens (Sec. 4 says "a set of masks"; already noted in factcheck-claims.md).
- Panel 6: N = 6 recent memories (claim sam2-n-recent); occlusion head (sam2.1_arxiv.tex:300, 968).
- Panel 7: concept prompt = noun phrase and/or exemplar, all instances (sam3/latex/main.tex:290); DETR-based
  detector with object queries (main.tex:353, 407-409); final score = p(match | present) * p(present)
  (main.tex:1433-1436).
- Panel 8: Boxes2Pixels mIoU_anom 0.6523 vs SegFormer-B2 0.5881 with "identical box-only supervision via
  SAM pseudo-masks" (boxes2pixels/latex/sec/4_expirements.tex:39-51, 57); 6.20 ms on an H100
  (4_expirements.tex:167, 174); SAM used offline only (3_method.tex:51).
- hero.html: 1,447 records, scan date 2026-10-09 (notes/arxiv-scan-2026-10-09.json: 1447 records).
- dialogs.html: arXiv versions match sources/*/ID (SAM v1, SAM 2 v2, SAM 3 v2, SAM 3D v2, etc.); image encoder
  64x64x256 at 1024x1024; stability score definition (segany.tex:675); occlusion head; presence product.
- chips.html: labels only, no facts.
