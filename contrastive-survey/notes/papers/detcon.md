# DetCon: Efficient Visual Pretraining with Contrastive Detection

## Identity
- arXiv 2103.10957, version read: v2 (v1 Fri 19 Mar 2021; v2 Thu 5 Aug 2021). Olivier J. Henaff ... Joao Carreira (6 authors).
- Venue: abs.html comments "Technical report" (PDF uses an ICCV template; venue not stated in source).

## Key idea
Contrast object-level features instead of whole-image features: pool features inside unsupervised masks
(Felzenszwalb-Huttenlocher by default), and treat the same mask in the other view as the positive, other masks and
other images as negatives. Many signals per image -> far less pretraining for the same transfer.

## Defining equation (src/detcon.tex, eq:con and eq:detcon2)
```
\ell_{\vm, \vm'} = - \log \frac{\exp( \vv_\vm {\cdot} \vv'_{\vm'})}{\exp(\vv_\vm {\cdot} \vv'_{\vm'})+\sum_\vn \exp(\vv_\vm {\cdot} \vv_{\vn})}
\mathcal{L} = \sum_{\vm} \sum_{\vm'} \mathds{1}_{\vm, \vm'} \ell_{\vm, \vm'}
```
v = mask-pooled projections, rescaled to norm 1/sqrt(tau), tau = 0.1; 16 masks sampled per image; indicator keeps
only pairs of masks covering the same region. DetCon_S uses SimCLR-style, DetCon_B BYOL-style training.

## Key numbers
- COCO detection (Mask R-CNN FPN, fine-tune 12 epochs "1x"), ResNet-50 pretrained on ImageNet ("fully-trained"; epochs
  not given in Table 2): DetCon_B AP^bb 42.7 vs BYOL 41.6, SimCLR 39.7, supervised 39.6 (Table 2, Fine-tune 1x col).
- Efficiency: "1,000 epochs of SimCLR pretraining is surpassed by only 200 epochs of DetCon_S" = "5x gain" (Sec. 4.1, COCO detection).
