# Chopra, Hadsell, LeCun 2005: siamese similarity metric

## 1. Identity
- Title: Learning a Similarity Metric Discriminatively, with Application to Face Verification
- Venue: CVPR 2005. Non-arXiv; read sources/chopra2005/paper.pdf (8 pages).
- Authors: Sumit Chopra, Raia Hadsell, Yann LeCun (NYU Courant).

## 2. Core problem
Face verification where categories are many, unknown at training time, with few samples each.
Learn a mapping so a simple distance in the output space approximates semantic distance.

## 3. Prior methods and why that hurt
Eigenfaces (PCA), Fisherfaces (LDA): linear and "very sensitive to geometric transformations"; hand-designed
invariances (tangent distance, elastic matching) (Sec 1.1). Classifiers need all categories at training time (Sec 1).

## 4. Key idea
A siamese conv net with shared weights maps each face to a point; train the energy E_W = ||G_W(X1) - G_W(X2)||
to be low for genuine pairs and high for impostor pairs. A contrastive term is required or the net collapses.

## 5. Mechanism
1. Pairs from the same person (genuine, Y=0) or different persons (impostor, Y=1).
2. Both images through the same G_W (conv net C1-S2-C3-S4-C5-F6), F6 has 50 units (Sec 3.2).
3. Energy E_W = norm of the embedding difference (Eq. 1); Sec 2.3 contrasts it with "the square norm" and calls it "the L1 norm".
4. Loss Eq. 8-9: increases with genuine energy, decreases with impostor energy.
5. Verification: Gaussian model per subject over embeddings; threshold the genuine probability (Sec 4.1).

## 6. Equations (PDF pp.3-4, transcribed)
Eq. (1): `E_W(X_1,X_2) = \|G_W(X_1) - G_W(X_2)\|`
Condition 1 (Sec 2.2): `\exists m > 0, such that E_W(X_1,X_2) + m < E_W(X_1,X_2')` -- "The positive number m can be interpreted as a margin."
Eq. (8)-(9), "The exact loss function that we use for a single sample":
```
L(W,Y,X_1,X_2) = (1-Y) L_G(E_W) + Y L_I(E_W)
               = (1-Y)\frac{2}{Q}(E_W)^2 + (Y) 2Q e^{-\frac{2.77}{Q} E_W}
```
- Y = 0 genuine, Y = 1 impostor. L_G: genuine partial loss. L_I: impostor partial loss.
- Q: "set to the upper bound of E_W" (components of G_W are bounded).
- The margin m is in the design condition, not in the loss: there is no hinge.
Collapse statement (Sec 2): minimizing E_W over genuine pairs only "generally leads to a catastrophic
collapse: The energy and the loss can be made zero by simply making G_W(X1) a constant function."
Sec 2.3 remark: the L1-type norm is used for energy, not the squared norm, because a squared norm's
gradient vanishes near zero energy ("a dangerous plateau").

## 7. Setup
| item | value |
|---|---|
| datasets | AT&T (400 images, 40 subjects, 112x92 -> 56x46); AR/Purdue (3,536 images, 136 subjects) + Feret subset (1,122 images, 187 subjects) |
| train pairs | AR/Feret: 140,000 pairs, half genuine half impostor |
| net | C1 15 maps 7x7; S2; C3 45 maps 6x6; S4; C5 250 maps 5x5; F6 50 units |
| optimizer | stochastic gradient |

## 8. Results
False reject % at given false accept % (Table 1, lower): AT&T test 0.00 / 1.00 / 1.00 at FA 10% / 7.5% / 5%;
AR test 11 / 14.6 / 19.

## 9. Limitations
Pair-wise only; hand-set constants (Q, 2.77); no margin hinge. DrLIM 2006 moves to the hinge-margin form.

## 10. Outline feedback
- The outline calls Chopra 2005 and the triplet loss "the same idea with a margin". For Chopra the
  margin appears only in Condition 1; the actual loss (Eq. 9) is quadratic + exponential with no hinge.
  Suggest: "Chopra 2005 introduced the siamese pull/push energy; DrLIM 2006 gave it the margin hinge."
