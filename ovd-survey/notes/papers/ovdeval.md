# OVDEval

## 1. Identity
- Title: How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection
- arXiv 2308.13177, read v2. v1 2023-08-25; v2 2023-12-18 (abs.html).
- Authors: Yiyang Yao (Northwestern Polytechnical University) ... Qing Wang (NWPU). Eight authors; Tiancheng Zhao and Kyusong Lee (Binjiang Institute of Zhejiang University) among them.
- Venue: AAAI 2024 (abs.html comment).
- Repo/data: github.com/om-ai-lab/OVDEval.

## 2. Core problem
COCO, LVIS, Flickr30K and RefCOCO test object types and referring expressions, but not attributes, positions, relations, negation or world knowledge, and REC/grounding data assume the phrase is present in the image (no hard negatives). Also, plain COCO AP can be gamed when labels are fine-grained.

## 3. What prior benchmarks did and why that hurt (per the paper)
- No systematic probing of fine-grained language ("Lack of systematic probing", Sec. 1).
- No hard negatives: grounding/REC pair the text with the image, so the model never has to reject a near-miss description (Sec. 1).
- COCO AP matches predictions to ground truth only within the same label, so a model that puts every candidate label on every object loses little (the Inflated AP Problem, Sec. "The Inflated AP Problem").

## 4. The one key idea
Test each linguistic aspect separately with hand-built hard negatives (the same object with a different color, material, verb, position word, or with "not" removed), and score with NMS-AP: run class-ignored NMS on predictions that overlap a ground-truth box before computing AP, so only the single most confident label per object counts.

## 5. Mechanism, step by step
1. Nine sub-datasets in six aspects (Table 1): Object (COCO val 2017, 80 classes), Proper noun (Logo, Landmark, Celebrity), Attribute (Color, Material), Position, Relationship, Negation.
2. Images: COCO val 2017 for Object; HICO for Relationship (top interaction label removed, repeated 10 times; active and passive phrasing); Visual Genome + Oscar labels for Color (6 objects x 6 colors = 36 combinations); LAION-400M retrieved with CLIP for the rest, with "multiple" added to queries for Position and Negation (Sec. "Image Collection").
3. Hard negatives per aspect (Sec. "Hard Negative Labels"): Color -> same object, other colors; Material -> same object, other materials; Relationship -> same subject and object, other verb; Position -> swap the position word (left -> right, above, under, front, back, in); Negation -> the positive label with "not" removed. Example (Fig. 2): positive "person who is not wearing a black shirt", negative "person wearing a black shirt".
4. Expert annotators label all boxes; ambiguous images are dropped (Sec. "Manual Annotation").
5. Score with NMS-AP (Algorithm 1): for each ground-truth box k, collect predictions with IoU > 0.5 to k, keep only the top-confidence one among them (class-ignored NMS, C-NMS), keep all other predictions, then compute standard COCO AP.

## 6. Equations
Two displayed, numbered equations (PDF Eq. 1 and 2; unnumbered `equation` environments in main.tex, AAAI style numbers them). They illustrate the inflated-AP toy case, not a model.
- Eq. (1): `Precision = \frac{TP}{TP+FP} = \frac{1}{1+1} = 0.50`
- Eq. (2): `Recall = \frac{TP}{GTnum} = \frac{1}{1} = 1.0`
- Paper's scenario: one image, two ground-truth objects labeled "red car" and "blue car". A model that does not understand color predicts 4 boxes, two per object, each object getting both labels, all with IoU > 0.95. Per class: one TP, one FP, one GT. So AP per class = 0.50 and mAP = 0.50 "even though it only detects the target objects without comprehending their descriptions."
- Gloss: TP = predictions with the right label matched to a GT box; FP = predictions with that label that match nothing of that label; GTnum = ground-truth instances of that label in the image.
- Our reading of the toy: the AP of 0.50 depends on the wrong-label box ranking below the right-label box for half the classes; the paper does not state scores. The point is that wrong labels on the right box are never charged against the right label.

## 7. Measurement setup (analysis lens)
- Claim 1: current OVD models only work on plain object names. Measured: NMS-AP per sub-dataset for six models (Table 3); radar chart (Fig. 4).
- Claim 2: plain AP overstates fine-grained understanding. Measured: AP and NMS-AP side by side on the same predictions (Table 3). Falsified if NMS-AP were close to AP on the hard-negative sets. It holds on COCO (GDINO 52.50 vs 55.30) and fails on Relationship (10.70 vs 35.30).
- Claim 3: models put several mutually exclusive labels on one object, and the top-scoring one is often wrong (Fig. 3, App. figures). Supporting measure: score histograms of positive vs negative labels for predictions with IoU > 0.9 to GT, Object vs Negation (Fig. 5, App.).
- Models (Table 2): Detic Swin-B (IN-21K, COCO, LVIS), 141.6M; MDETR R101 (VG, Flickr30k, COCO), 185M; GLIP Swin-L (FourODs, GoldG, CC3M+12M, SBU), 430.42M; FIBER Swin-B (Flickr30k, MixedNoCOCO, O365), 252.06M; OmDet ConvNext-B (O365, GoldG, PhraseCut, HOI-A, VAW, RefCOCO), 241.5M; Grounding DINO Swin-B (COCO, O365, GoldG, Cap4M, OpenImage, ODinW-35, RefCOCO), 232.9M.
- Not reported: OWL-ViT and OWLv2 are not evaluated.
- Prompt format per model and score thresholds are not stated.
- Dataset size (Table 1): images per sub-dataset 1,170 (Color) to 5,000 (COCO); labels 9 (Logo, Landmark), 10 (Celebrity), 36 (Color), 90 (Material), 319 (Relationship), 2,414 (Negation), 7,301 (Position); average negative labels per positive 1.00 (Negation) to 9.00 (Celebrity); average label words 1.10 (COCO) to 9.67 (Position).

## 8. Results worth remembering
Table 3, NMS-AP / AP in %. Starred COCO values (Grounding DINO, Detic) are supervised, not zero-shot.
| Sub-dataset | GLIP | FIBER | Grounding DINO | Detic | MDETR | OmDet |
|---|---|---|---|---|---|---|
| COCO | 48.90 / 51.30 | 46.80 / 49.30 | 52.50* / 55.30* | 45.30* / 45.80* | 1.60 / 3.20 | 54.68 / 57.50 |
| Logo | 10.20 / 17.61 | 6.30 / 9.05 | 10.30 / 14.60 | 9.60 / 9.60 | 0.90 / 4.60 | 6.10 / 11.00 |
| Landmark | 20.30 / 36.36 | 11.00 / 16.99 | 15.10 / 23.40 | 30.00 / 30.08 | 1.80 / 7.80 | 26.30 / 32.38 |
| Celebrity | 4.60 / 8.24 | 0.80 / 3.31 | 0.70 / 2.00 | 0.00 / 0.00 | 1.10 / 4.80 | 1.80 / 6.36 |
| Proper noun avg | 11.70 / 20.74 | 6.03 / 9.78 | 8.70 / 13.33 | 13.20 / 13.23 | 1.27 / 5.73 | 11.40 / 16.58 |
| Color | 3.70 / 6.70 | 6.80 / 9.40 | 9.40 / 12.41 | 3.90 / 4.14 | 3.10 / 7.30 | 22.90 / 24.56 |
| Material | 7.40 / 15.87 | 12.40 / 17.72 | 9.00 / 15.50 | 9.20 / 9.75 | 2.50 / 10.70 | 16.30 / 22.59 |
| Attribute avg | 5.55 / 11.28 | 9.60 / 13.56 | 9.20 / 13.96 | 6.55 / 6.94 | 2.80 / 9.00 | 19.60 / 23.58 |
| Position | 30.90 / 48.10 | 34.30 / 48.20 | 67.50 / 77.40 | 12.20 / 14.40 | 34.00 / 48.80 | 21.20 / 47.75 |
| Relationship | 10.00 / 33.20 | 14.50 / 31.40 | 10.70 / 35.30 | 6.10 / 7.20 | 8.20 / 29.40 | 41.98 / 51.98 |
| Negation | 29.30 / 51.80 | 28.70 / 57.20 | 52.50 / 67.30 | 27.90 / 29.70 | 28.30 / 41.10 | 35.10 / 55.86 |
| Total average | 18.37 / 29.91 | 17.96 / 26.95 | 25.30 / 33.69 | 16.02 / 16.74 | 9.06 / 17.52 | 25.86 / 39.15 |

- Attribute is the weakest aspect for every model except where pre-training had attribute data: OmDet used VAW and leads Color (22.90) and Material (16.30) (Sec. "Main Results").
- Grounding DINO Position 67.50 is attributed to RefCOCO in its pre-training; OmDet Relationship 41.98 to HOI-A (Sec. "Main Results").
- Celebrity NMS-AP is "close to 0%" for all (0.00 to 4.60).
- Inflation: Grounding DINO Relationship 35.30 AP -> 10.70 NMS-AP; GLIP Negation 51.80 -> 29.30; COCO barely moves (55.30 -> 52.50). Intro: "the conventional AP score is 30% higher than the model's actual performance" (no exact computation given).
- Detic shows the smallest AP vs NMS-AP gap; the paper says Detic already applies NMS so it rarely stacks labels on one box (Sec. "Comparing NMS-AP").

## 9. Limitations the paper states, and failure cases
- No explicit limitations section.
- Failure cases: GLIP predicts "cat that is sitting" and "cat that is not sitting" on the same cat (Fig. 3, Sec. "Comparing NMS-AP"). Predictions with the highest confidence are "frequently incorrect", worst for models with many output boxes such as Grounding DINO (Sec. "Limitations of Current OVD Models").
- In GLIP's score histogram, positive and negative labels separate on Object but overlap on Negation, and Negation predictions sit mostly at low confidence (Fig. 5).
- Text-table mismatch: Sec. "Statistics" says 20K images with "3K meticulously annotated labels"; Table 1 lists 7,301 Position labels and 2,414 Negation labels alone. Record as printed.

## 10. Outline feedback
- Supports (ch 13 item 5): attributes, relations and negation fail. Best zero-shot NMS-AP on Color is 22.90 (OmDet, attribute pre-training) and 9.40 for Grounding DINO; on Relationship 10.70 for Grounding DINO. This is the strongest evidence for "an adjective is not a prompt feature you can rely on". Industrial relevance (our reading): defect classes are often attribute or material words ("rusted bolt", "cracked weld"); OVDEval says those modifiers are mostly ignored.
- Supports (ch 13 item 4, calibration): score histograms overlap on Negation; top-confidence labels are often wrong.
- Supports a measurement point ch 12 needs: when the candidate labels are near-synonyms, plain AP overstates the model; evaluate with one label per object (NMS-AP-like) or with a fixed operating threshold.
- Does NOT cover OWL-ViT/OWLv2 at all. The outline names only "OVDEval, D3" for ch 13 item 5; do not attribute OVDEval numbers to OWL models.
- Grounding DINO here is Swin-B with RefCOCO and ODinW-35 in pre-training, and its COCO score is supervised (starred). Do not compare its OVDEval numbers to the Swin-T zero-shot numbers of ch 9 without saying so.
- Proper-noun results (logo, landmark, celebrity) speak to "world knowledge", not to industrial parts; keep them out of the industrial argument except as "names the model cannot ground".
- Practical question, plainly: OVDEval supports "a prompt that relies on an attribute, a relation or a negation will mostly be read as the bare noun". It does not measure rare or industrial objects (images are COCO, Visual Genome, HICO and LAION web images), so it says nothing direct about when a prompt can replace a custom detector for a factory part.
