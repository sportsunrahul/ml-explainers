# FaceNet: A Unified Embedding for Face Recognition and Clustering (branch, triplet loss in full)

## Identity
- arXiv 1503.03832, read v3 (2015-06-17); v1 2015-03-12. Authors: Florian Schroff, Dmitry Kalenichenko, James Philbin (Google). Venue: CVPR 2015 (DOI 10.1109/CVPR.2015.7298682 in abs.html).

## Key idea
Train a CNN to output a 128-d unit vector per face so that squared L2 distance means face
similarity, using a triplet loss with a margin alpha and online semi-hard negative mining.

## Equations (Sec 3.1, PDF Eq. 1 and 3; facenet.tex labels `eq:triplet_constraint`, `eq:triplet_loss`)
Constraint, Eq. (1):
```
\|f(x_i^a) - f(x_i^p)\|_2^2 + \alpha &< \|f(x_i^a) - f(x_i^n)\|_2^2\;,\\
\:\forall\left(f(x_i^a), f(x_i^p), f(x_i^n)\right)\in\mathcal{T}\;.&&
```
Loss, Eq. (3), "The loss that is being minimized is then L =":
```
\sum_i^N\left[\left\|f(x_i^a)-f(x_i^p)\right\|_2^2 - \left\|f(x_i^a)-f(x_i^n)\right\|_2^2+\alpha\right]_+\;.
```
- x_i^a: anchor face. x_i^p: another face of the same person (positive). x_i^n: a face of another person (negative).
- f(x): embedding with ||f(x)||_2 = 1 (on the hypersphere).
- alpha: "a margin that is enforced between positive and negative pairs". [.]_+ = max(0, .).
- T: set of all possible triplets, cardinality N.
- Grades: triplets that violate the margin; satisfied triplets give zero loss.
Semi-hard negatives, Eq. (4): `\left\|f(x_i^a)-f(x_i^p)\right\|_2^2<\left\|f(x_i^a)-f(x_i^n)\right\|_2^2` -- "Those negatives lie inside the margin alpha."
Hardest negatives "can ... result in a collapsed model (i.e. f(x)=0)" (Sec 3.2).

## Setup
- Margin alpha = 0.2 (Sec 3.3, "The margin alpha is set to 0.2").
- Batch about 1,800 exemplars; about 40 faces per identity per batch (Sec 3.2). All anchor-positive pairs, semi-hard negatives.
- SGD + AdaGrad, lr 0.05; CPU cluster 1,000 to 2,000 hours (Sec 3.3). Embedding 128-d.

## Key number
LFW accuracy 99.63 (+-0.09) with face alignment; 98.87 with fixed center crop (Sec 5.6 text; 99.63 also in abstract).

## Outline feedback
Fine as cited. Note squared distances on unit vectors: alpha = 0.2 is in squared-distance units (range 0..4).
