# Hadsell, Chopra, LeCun 2006: DrLIM

## 1. Identity
- Title: Dimensionality Reduction by Learning an Invariant Mapping
- Venue: CVPR 2006 (PDF header: "November 2005. To appear in CVPR 2006"). Non-arXiv; read sources/hadsell2006/paper.pdf (8 pages).
- Authors: Raia Hadsell, Sumit Chopra, Yann LeCun (NYU Courant).

## 2. Core problem
Learn a function G_W that maps high-dim inputs to a low-dim space where Euclidean distance
reflects given neighbor relations, works on unseen points, and can learn invariances.

## 3. Prior methods and why that hurt
LLE, Laplacian eigenmaps, ISOMAP, MDS: need a meaningful input-space metric and/or cannot map new
points without recomputation (Sec 1.1). LLE gives degenerate embeddings on shifted digits (Fig. 6) and
organizes NORB by lighting (Fig. 10).

## 4. Key idea
A pair loss: pull similar pairs together (attract-only spring), push dissimilar pairs apart only
while they are closer than a margin m (m-repulse-only spring). The push term prevents collapse.

## 5. Mechanism
1. Build pairs from prior knowledge (k nearest neighbors, known translations, temporal adjacency): Y=0 similar, Y=1 dissimilar.
2. Siamese net: two copies of G_W with shared W; D_W = ||G_W(X1) - G_W(X2)||_2.
3. Loss Eq. 4; SGD; gradient is the sum from both copies (Sec 3.1).

## 6. Equations (PDF p.3, Sec 2.1-2.2; transcribed from PDF)
Eq. (1): `D_W(\vec{X}_1,\vec{X}_2) = \|G_W(\vec{X}_1) - G_W(\vec{X}_2)\|_2`
Eq. (3): `L(W,(Y,\vec{X}_1,\vec{X}_2)^i) = (1-Y) L_S(D_W^i) + Y L_D(D_W^i)`
Eq. (4) "The exact loss function":
```
L(W,Y,\vec{X}_1,\vec{X}_2) = (1-Y)\frac{1}{2}(D_W)^2 + (Y)\frac{1}{2}\{max(0, m - D_W)\}^2
```
- Y = 0 for a similar pair, Y = 1 for a dissimilar pair (later papers often flip this).
- D_W: Euclidean distance between the two embeddings.
- m > 0: margin; "defines a radius around G_W(X). Dissimilar pairs contribute to the loss function
  only if their distance is within this radius" (Sec 2.1).
- Grades: every labelled pair; similar pairs always; dissimilar pairs only when D_W < m.
Eq. (6) L_S = 1/2 D_W^2; Eq. (8) L_D = 1/2 (max{0, m - D_W})^2; Eq. (9) gradient of L_D is
-(m - D_W) dD_W/dW for D_W < m, zero for D_W > m.
Collapse statement (Sec 2.1): "Simply minimizing D_W(X1,X2) over the set of all similar pairs will
usually lead to a collapsed solution, since D_W and the loss L could then be made zero by setting G_W to a constant."

## 7. Setup
| experiment | G_W | output dim | pairs |
|---|---|---|---|
| MNIST 4s and 9s (Sec 3.2) | conv net C1(15 maps, 6x6)-S2-C3(30 maps, 9x9)-F(2 units) | 2 | 5 NN per sample: 30,000 similar, ~18 million dissimilar |
| Shifted MNIST (Sec 3.3) | same | 2 | shifts -6,-3,3,6 px; 30,000 samples |
| NORB airplane (Sec 3.4) | 2-layer FC, 20 hidden | 3 | 972 images; 660 train / 312 test; 10,989 similar, 206,481 dissimilar |
- Margin value m: NOT STATED for any experiment. Figure 1 marks "margin: m" at energy 1.25 on its
  x-axis; this is an illustrative plot value only.
- Optimizer: SGD; lr, epochs not stated.

## 8. Results (qualitative only)
- Fig. 4: test digits spread "evenly" on 2D. Fig. 7: shift-invariant 2D map. Fig. 8: NORB maps to a
  cylinder (azimuth around, elevation along), invariant to lighting. No accuracy tables.

## 9. Limitations
Needs explicit pair labels from prior knowledge; quadratic number of dissimilar pairs (P "may be as
large as the square of the number of samples"). Margin must be hand-picked. Softmax losses (CPC/InfoNCE) remove the margin.

## 10. Outline feedback
- "Eq. 4" is correct. Draw m as a ring: correct per Sec 2.1 ("radius around G_W(X)").
- No paper margin value for the toy; if the toy needs one, the only printed number is Fig. 1's 1.25
  (illustrative). Label any toy margin as our choice.
- The spring analogy (Sec 2.2, Fig. 2) maps directly onto viz-pullpush and the hero's green/red forces.
