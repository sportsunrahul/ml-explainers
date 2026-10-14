# InfoMin (branch)

## Identity
- Title: What Makes for Good Views for Contrastive Learning?
- arXiv 2005.10243, read v3. v1 submitted 2020-05-20; v3 revised 2020-12-18 (abs.html).
- Authors: Yonglong Tian ... Phillip Isola (Tian, Sun, Poole, Krishnan, Schmid, Isola). Style file `neurips_2020.sty`.

## Key idea (two sentences)
Good views share only the information the downstream task needs: reduce the mutual information between views while
keeping task-relevant information ("InfoMin principle"). Accuracy vs I_NCE between views traces a reverse-U with a
sweet spot; stronger augmentation moves toward it (e.g. RandomResizedCrop low bound c = 0.2, color jitter 1.0).

## Defining statement (Proposition 3.1, `\label{theory:infomin}`, Sec. 3)
```
(\mathbf{v_1}^*, \mathbf{v_2}^*) = \argmin_{\mathbf{v_1}, \mathbf{v_2}} I(\mathbf{v_1}; \mathbf{v_2}),
\text{ subject to } I(\mathbf{v_1}; \mathbf{y}) = I(\mathbf{v_2}; \mathbf{y}) = I(\mathbf{x}; \mathbf{y})
```
(Condition: f_1, f_2 minimal sufficient encoders; task T with label y. Transcribed from the prop text.)

## Key number
- InfoMin Aug, ResNet-50, MLP head, 800 epochs: 73.0 top-1 linear (Table 1, last row); 200 epochs: 70.1.
  Pretraining in the MoCo framework (65536 negatives, batch 256, 8 GPUs for the ablation plots; App. "Details").
- Crop sweet spot c = 0.2 vs the common 0.08; 0.08 costs > 1% with a nonlinear head (App., Cropping).
