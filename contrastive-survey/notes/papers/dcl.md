# DCL (Yeh et al. 2022) -- Decoupled Contrastive Learning [branch]

- Title: "Decoupled Contrastive Learning". arXiv 2110.06848, read v3 (2022-07-30); v1 2021-10-13 (abs.html).
- Authors: Chun-Hsiao Yeh ... Yann LeCun (with Cheng-Yao Hong, Yen-Chi Hsu, Tyng-Luh Liu, Yubei Chen). Venue: ECCV 2022 (abs.html comment).

Key idea: the InfoNCE gradient carries a "negative-positive coupling" multiplier q_B = 1 - (positive softmax probability), Eq. (4); with small batches the task is easy, q_B drops toward 0, and learning slows. Removing the positive term from the denominator removes q_B, which makes the loss far less sensitive to batch size.

Defining equation, Eq. (5)-(6) (label `eq:DC`, printed number (6) on the second line), Sec. 3:
```
L_{DC,i}^{(k)} &= -\log{\frac{\exp(\langle \mathbf{z}_i^{(1)},\mathbf{z}_i^{(2)} \rangle/\tau)}{\bcancel{\exp(\langle \mathbf{z}_i^{(1)},\mathbf{z}_i^{(2)} \rangle/\tau)} + U_{i,k}}} \\
&= -\langle \mathbf{z}_i^{(1)},\mathbf{z}_i^{(2)} \rangle/\tau + \log{U_{i,k}}
```
U_{i,k} (Eq. 2) = sum over the other 2N-2 views of exp(<z_i^(k), z_j^(l)>/tau).

Key number: ImageNet-1K linear top-1, ResNet-50, batch 256, 200 epochs: SimCLR 61.8 -> SimCLR+DCL 65.9 (Sec. 4, Fig. 3 text); with tuned tau/lr and BYOL asymmetric augmentation 68.2 (Table 3; abstract "outperforming its SimCLR baseline by 6.4%"). Fig. 3: tau 0.1 for SimCLR, 0.07 for MoCo; MoCo with 256 queue 54.7 -> 60.8.

Note: Appendix 5 argues Wang & Isola's L_uniform has "negative-negative coupling".

Gradient print: Eq. (3) (`eq:gradient_Li`, Proposition 1) prints the full InfoNCE gradient wrt z_i^(1), z_i^(2) and each negative z_j^(l), with per-negative softmax weight exp(<z_i^(1),z_j^(l)>/tau)/U_{i,1} and the shared NPC multiplier q_B (Eq. 4). Useful cross-check for the page's own derivation (alongside Wang & Liu Eq. 4, which is wrt similarities).
