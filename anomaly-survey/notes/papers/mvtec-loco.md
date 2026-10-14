# MVTec LOCO AD (SECONDARY SOURCES ONLY)

Every fact here comes from EfficientAD (arXiv 2303.14535v3) or MVTec AD 2 (arXiv 2503.21622v1). The primary paper (Bergmann et al., "MVTec LOCO", IJCV 2022, cited as `bergmann2021_mvtec_loco_ijcv`) is NOT in `sources/` (`sources/mvtec-loco/` is empty). All claims carry `"secondary_source": true`.

## 1. Identity
- Name as cited: "MVTec Logical Constraints (MVTec LOCO) dataset" (EfficientAD Sec. 2.1, p. 2); "MVTec LOCO AD" (MVTec AD 2 Sec. 2.1, Table 2).
- Primary paper: not available. arXiv id, date, authors list, image counts: not available from the secondary sources.
- The GCAD method is introduced in the LOCO paper (EfficientAD Sec. 2.2, p. 3: "GCAD [LOCO] lets an autoencoder reconstruct images in the feature space of a pretrained network").

## 2. Core problem (as the secondary sources state it)
Structural anomalies (stains, scratches) are local and visible in a patch. Logical anomalies are "violations of logical constraints, for example, a wrong ordering or a wrong combination of normal objects" (EfficientAD Sec. 2.1). In MVTec AD 2's words: "elements in a defective image maintain the same general appearance but violate logical constraints. For example, this might include an incorrect number of elements in a box" (MVTec AD 2 Sec. 2.1).

## 3. What prior datasets did and why that hurt
MVTec AD and VisA have no logical defects (MVTec AD 2 Table 2: "Logical Defects" is checked only for MVTec LOCO AD).

## 4. The one key idea
A benchmark in the MVTec AD format with both structural and logical anomalies, scored separately and then averaged.

## 5. Facts available
1. Five scenarios: "VisA and MVTec LOCO follow the design of MVTec AD and comprise twelve and five anomaly detection scenarios, respectively" (EfficientAD Sec. 2.1, p. 2).
2. Each scenario is a separate dataset with train and test sets; pixel-precise masks (EfficientAD Sec. 2.1).
3. Train and validation sets contain no anomalous images (EfficientAD Sec. 4, p. 6).
4. Scenario names named in the secondary text (not as a list):
   - splicing connectors: EfficientAD Fig. 5 caption (p. 5): "Normal input images contain a horizontal cable connecting the two splicing connectors at an arbitrary height"; structural example = a small metal washer at the cable end; logical example = a second cable.
   - (banana) juice bottle, breakfast box, box of pushpins: EfficientAD App. F text (p. 24), referring to Fig. 9 (p. 25).
   - The fifth scenario is NOT named in either secondary source. Do not fill it in from memory; get the IJCV PDF.
5. Logical anomaly types listed by EfficientAD (Sec. 3.3, p. 5): "missing, misplaced, or surplus objects or the violation of geometrical constraints, for example, the length of a screw." Conclusion (Sec. 5, p. 8): "Fine-grained logical anomalies, however, remain a challenge -- for example a screw that is two millimeters too long."
6. Metric: "For MVTec LOCO, we use the AU-sPRO metric [LOCO], a generalization of the AU-PRO metric for evaluating the localization of logical anomalies" (EfficientAD Sec. 4, p. 6). The sPRO definition itself is NOT in either secondary source.
7. Evaluation weighting: "For MVTec LOCO, we use the official evaluation script, which gives logical and structural anomalies an equal weight in the computed metrics" (EfficientAD Sec. 4, p. 6). App. D (p. 19): each metric is computed separately on structural and on logical anomalies, then the two are averaged per scenario.

## 6. Equations
None available. The sPRO equation is in the IJCV paper only.

## 7. Measurement setup
- Image counts, train/test split sizes, resolutions: NOT available in secondary sources.
- EfficientAD resizes all LOCO images to 256 x 256 and maps the anomaly map back to the original size (App. A.1, p. 14).

## 8. Results worth remembering (EfficientAD numbers on LOCO, %, means over the 5 scenarios)
- Image AU-ROC, EfficientAD Table 2 (p. 7) / Table 10 (p. 19): EfficientAD-M logical 86.8, structural 94.7, LOCO mean 90.7. PatchCore logical 75.8, structural 84.8, mean 80.3. GCAD logical 83.9, structural 82.7, mean 83.3. Protocol: one model per scenario, 256 x 256; EfficientAD = mean of 5 runs.
- Localization AU-sPRO up to FPR 0.3, EfficientAD Table 12 (p. 20): EfficientAD-M LOCO mean 92.5 (structural 93.7, logical 91.3); PatchCore mean 70.4; GCAD 89.5.
- Localization up to FPR 0.05, EfficientAD Table 13 (p. 20): EfficientAD-M LOCO mean 79.8 (structural 83.2, logical 76.5). This is the "79.8% in localization performance" MVTec AD 2 Sec. 2.1 quotes for the "currently best model" on LOCO (MVTec AD 2 does not name the FPR limit; the match is our reading).
- Logical anomalies are harder for local methods: every method except GCAD scores lower on logical than on structural image AU-ROC in EfficientAD Table 2 (our reading of the table).

## 9. Limitations (per secondary sources)
- Fine-grained logical anomalies (a screw 2 mm too long) remain hard (EfficientAD Sec. 5, p. 8).
- MVTec AD 2 deliberately excludes logical anomalies (Table 2 caption).

## 10. Outline feedback
- The outline's plan "LOCO facts come from EfficientAD and MVTec AD 2" covers: five scenarios, logical vs structural, sPRO's name, and the equal-weight averaging. It does NOT cover: the image counts, the fifth scenario name, or the sPRO definition. Get the IJCV PDF into `sources/mvtec-loco/` before the page states counts or explains sPRO.
- Until then, the page should describe sPRO only as "a generalization of AU-PRO for logical anomalies" (EfficientAD Sec. 4) and give no formula.
