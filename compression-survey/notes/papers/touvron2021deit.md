# touvron2021deit -- Training data-efficient image transformers & distillation through attention (BRANCH)

## Identity
- arXiv 2012.12877, version read: v2 (2021-01-15). v1 submitted 2020-12-23 (abs.html).
- Authors: Hugo Touvron ... Herve Jegou (6 authors). Venue: not stated in abs.html or the v2 TeX.
- Repo: paper says "We share our code and models" (Sec. 1); no URL downloaded.
- Custom macros in the TeX (`src/main.tex`): `\ours` = DeiT, `\oursbase` = DeiT-B, `\oursbasedis` = DeiT-B
  plus an alembic icon (`\alambic`, an image) marking the distilled model; `\pzo` = `\phantom{0}`.

## Key idea (two sentences)
Add a learned "distillation token" next to the class token of a ViT; it attends with the patch tokens through
all layers, and its output head is trained to reproduce the teacher's prediction while the class head is trained
on the true label. With a convnet teacher (RegNetY-16GF) and hard-label distillation, the transformer beats both
its no-distillation version and classic soft distillation; at test time the two heads' softmax outputs are added.

## Equations (TeX verbatim, `src/token.tex`; numbers from PDF Sec. 4; Eq. 1 is attention)
Eq. 2, soft distillation (Hinton-style, written with tau^2 and KL):
```
\mathcal{L}_\mathrm{global} = (1-\lambda) \mathcal{L}_\mathrm{CE}(\psi(Z_\mathrm{s}),y) + \lambda  \tau^2 \mathrm{KL}(\psi(Z_\mathrm{s}/\tau),\psi(Z_\mathrm{t}/\tau)).
```
Gloss: Z_s, Z_t = student and teacher logits; psi = softmax; tau = temperature; lambda balances KL and CE on
true labels y. Argument order inside KL is as printed. Values used: "$\tau=3.0$ and $\lambda=0.1$ for the usual
(soft) distillation" (Sec. 6, training.tex).
Eq. 3, hard-label distillation:
```
\mathcal{L}_\mathrm{global}^\mathrm{hard Distill} =  \frac{1}{2}\mathcal{L}_\mathrm{CE}(\psi(Z_s),y) +  \frac{1}{2}\mathcal{L}_\mathrm{CE}(\psi(Z_s),y_\mathrm{t}).
```
with $y_\mathrm{t}=\mathrm{argmax}_c Z_\mathrm{t}(c)$. Paper: "this choice is better than the traditional one,
while being parameter-free and conceptually simpler: The teacher prediction $y_\mathrm{t}$ plays the same role as
the true label $y$." Label smoothing epsilon = 0.1 on true labels.
With the token: the distillation token's "target objective is given by the distillation component of the loss"
(Sec. 4). `our reading`: class head gets the y term, distillation head gets the y_t term of Eq. 3.
Teacher fixed: not stated in words; teacher is pre-trained ("we assume we have access to a strong image
classifier as a teacher model", Sec. 4). Fig. 2 caption: "Both the class and distillation tokens input to the
transformers are learned by back-propagation"; the paper says nothing about teacher updates. `our reading`:
teacher frozen.

## Teacher
RegNetY-16GF, 84M params, 82.9% ImageNet top-1, "trained with the same data and same data-augmentation as DeiT"
(Sec. 5.2). Table 2: convnet teachers beat a DeiT-B teacher (student 81.9 with DeiT-B teacher vs 83.1 with
RegNetY-16GF, pretrain at 224).

## Key rows
Table 3 (ImageNet top-1 %, 300 epochs; columns Ti 224 | S 224 | B 224 | B^384):
| method | label | teacher | Ti 224 | S 224 | B 224 | B^384 |
|---|---|---|---|---|---|---|
| DeiT -- no distillation | yes | no | 72.2 | 79.8 | 81.8 | 83.1 |
| DeiT -- usual distillation | no | soft | 72.2 | 79.8 | 81.8 | 83.2 |
| DeiT -- hard distillation | no | hard | 74.3 | 80.9 | 83.0 | 84.0 |
| DeiT (distilled): class embedding | yes | hard | 73.9 | 80.9 | 83.0 | 84.2 |
| DeiT (distilled): distil. embedding | yes | hard | 74.6 | 81.1 | 83.1 | 84.4 |
| DeiT (distilled): class+distillation | yes | hard | 74.5 | 81.2 | 83.4 | 84.5 |
(Label/teacher columns as printed: check and cross marks.)
Table 5 (ImageNet / Real / V2 top-1; throughput on one 16GB V100, largest batch per model, mean of 30 runs):
- DeiT-B: 86M, 224^2, 292.3 im/s, 81.8 / 86.7 / 71.5.
- DeiT-B (distilled): 87M, 224^2, 290.9 im/s, 83.4 / 88.3 / 73.2.
- DeiT-B (distilled) / 1000 epochs: 84.2. DeiT-B (distilled) at 384 / 1000 epochs: 85.8 im/s, 85.2.
- Teacher row RegNetY-16GF: 84M, 334.7 im/s, 82.9.

## Outline feedback
- The outline's loss "(1-a) CE + a T^2 KL" is DeiT Eq. 2 in form. DeiT uses lambda = 0.1 on the soft term,
  which is the opposite emphasis to Hinton's "lower weight" on the hard term.
- One-line delta: plain soft KD gave no gain on DeiT-B at 224 (81.8 -> 81.8); hard-label KD gave 83.0; the token
  plus late fusion gave 83.4. The student is a ViT and the teacher a convnet, so this is cross-architecture KD,
  not a smaller copy of the teacher (student 86-87M vs teacher 84M params).
