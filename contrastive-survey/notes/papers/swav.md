# SwAV (spine)

## 1. Identity
- Title: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
- arXiv 2006.09882, read v5 (2021-01-08). v1 2020-06-17.
- Authors: Mathilde Caron ... Armand Joulin. Venue: NeurIPS 2020 (abs.html comment; src uses `neurips_2020` final).
- Repo: github.com/facebookresearch/swav (Appendix A, first paragraph). No README in sources.
- Macros: `\OURS` = SwAV, `\resnetfifty` = ResNet-50, `\ImNet` = ImageNet.

## 2. Core problem
Contrastive instance methods compare every pair of features and need many negatives (big batch or a queue).
Clustering methods avoid pairwise comparison but are offline: they need a full pass over the dataset to assign clusters.
SwAV wants an online clustering method that works batch by batch.

## 3. Prior methods and why that hurt (per the paper)
- DeepCluster / SeLa: alternate "cluster the whole dataset" and "train on the codes"; not online, several passes needed (Sec 3).
- MoCo / SimCLR: pairwise comparisons; MoCo stores 65K features from the last 250 batches plus a momentum encoder (Sec 3.1 "Working with small batches").
- More crops help but compute and memory grow quadratically with crop count (Sec 3.2).

## 4. Key idea
Compute a soft cluster "code" for each view by assigning features to K learned prototypes, with an equipartition constraint
solved by 3 Sinkhorn-Knopp iterations; then predict the code of one view from the features of the other view ("swapped prediction").

## 5. Mechanism
1. Each image gives 2 full-res crops (224x224, scale 0.14-1) and V small crops (96x96, scale 0.05-0.14) (App A.2).
2. ResNet-50 -> 2048-d -> 2-layer MLP projection head -> 128-d -> L2-normalize -> z (App A.1; Sec 3.1).
3. Scores = z^T C, C = K = 3000 prototypes of dim 128 (App C.1; pseudocode "C: prototypes (DxK)"; scores 2BxK).
4. Codes q from the 2 full-res crops only, via Sinkhorn on the batch score matrix, under `torch.no_grad()` (App A.1 pseudocode). The codes are a target: no gradient flows through q.
5. p = softmax(z^T c_k / tau), tau = 0.1. Loss = cross-entropy between the code of crop i and p of every other crop v != i.
6. Backprop updates encoder and prototypes; after each step the prototypes are L2-normalized (pseudocode). Prototypes are frozen in epoch 1 (App A.1).
7. Small batches (256): a queue of 3,840 features (last 15 batches) augments Z for Sinkhorn only; starts at epoch 15 (Sec 4.3, App A.6).

Which positions are graded: for each image n, the code of crop t_i (i in {1,2}, full-res) grades the prediction p of every other crop of the same image.
No other image appears in the loss directly; other images enter only through the Sinkhorn equipartition (columns summed over the batch).

## 6. Equations
Eq. (1) `eq:twoviews`:
```
L(\mathbf{z}_t, \mathbf{z}_s) & = &\ell(\mathbf{z}_t, \mathbf{q}_s) + \ell(\mathbf{z}_s, \mathbf{q}_t),
```
"a swapped prediction problem". z_t, z_s: normalized features of two views; q_t, q_s: their codes; l: fit between features and a code.

Eq. (2) `eq:loss`:
```
\ell(\mathbf{z}_t, \mathbf{q}_s) = - \sum_{k} \mathbf{q}_s^{(k)} \log \mathbf{p}_t^{(k)}, \quad \text{where} \quad \mathbf{p}_t^{(k)} = \frac{ \exp \left ( \frac{1}{\tau} \mathbf{z}_t^\top \mathbf{c}_k \right ) }{\sum_{k'} \exp \left ( \frac{1}{\tau} \mathbf{z}_t^\top \mathbf{c}_{k'} \right ) }.
```
"the cross entropy loss between the code and the probability obtained by taking a softmax of the dot products of z_i and all prototypes in C".
q_s^(k): share of prototype k in view s's code; p_t^(k): softmax probability that view t belongs to prototype k; c_k: prototype k (unit vector); tau: temperature (0.1).

Full swapped-prediction loss (unnumbered, `\nonumber`, label `eq:swapploss2`):
```
  - \frac{1}{N} \sum_{n=1}^N \sum_{s,t \sim \mathcal{T}} \left [
  \frac{1}{\tau} \mathbf{z}_{nt}^\top \mathbf{C} \mathbf{q}_{ns}
  + \frac{1}{\tau} \mathbf{z}_{ns}^\top \mathbf{C} \mathbf{q}_{nt}
  - \log \sum_{k=1}^K \exp \left ( \frac{\mathbf{z}_{nt}^\top \mathbf{c}_k}{\tau} \right )
  - \log \sum_{k=1}^K \exp \left ( \frac{\mathbf{z}_{ns}^\top \mathbf{c}_k}{\tau} \right ) \right ] .
```
"jointly minimized with respect to the prototypes C and the parameters theta of the image encoder".

Eq. (3) `eq:assign` (code optimization):
```
\max_{\mathbf{Q}\in\mathcal{Q}} \ \text{Tr} \left( \mathbf{Q}^\top \mathbf{C}^\top \mathbf{Z} \right) +  \varepsilon H(\mathbf{Q}),
```
Q: K x B code matrix; Tr(Q^T C^T Z): total feature-prototype similarity of the assignment; H(Q) = -sum Q_ij log Q_ij: entropy; epsilon: smoothness.
Paper: high epsilon "generally leads to a trivial solution where all samples collapse into an unique representation"; "in practice we keep epsilon low".

Eq. (4) (transportation polytope on the minibatch):
```
\mathcal{Q} = \left \{\mathbf{Q}\in\mathbb{R}_+^{K\times B} ~|~\mathbf{Q} \mathbf{1}_B = \frac{1}{K} \mathbf{1}_K, \mathbf{Q}^\top \mathbf{1}_K = \frac{1}{B} \mathbf{1}_B \right \},
```
Rows (prototypes) each sum to 1/K, columns (images) each sum to 1/B. "on average each prototype is selected at least B/K times in the batch."

Eq. (5) `eq:qstar` (Sinkhorn solution):
```
  \mathbf{Q}^*= \text{Diag}(\mathbf{u}) \exp\left(\frac{\mathbf{C}^\top \mathbf{Z}}{\varepsilon} \right) \text{Diag}(\mathbf{v}),
```
u in R^K, v in R^B: renormalization vectors found by Sinkhorn-Knopp. Soft codes kept (no rounding): rounding "is a more aggressive optimization step than gradient updates".
"using only 3 iterations is fast and sufficient"; "alignment of 4K features to 3K codes takes 35ms".

Eq. (6) multi-crop loss (no label; number from PDF):
```
L(\mathbf{z}_{t_1}, \mathbf{z}_{t_2}, \dots, \mathbf{z}_{t_{V + 2}}) = \sum_{i \in \{1, 2\}} \sum_{v=1}^{V + 2} \mathbf{1}_{v \neq i} \ell(\mathbf{z}_{t_v}, \mathbf{q}_{t_i}).
```
"we compute codes using only the full resolution crops."

Sinkhorn pseudocode (App A.1, verbatim, for `viz-sinkhorn`):
```
def sinkhorn(scores, eps=0.05, niters=3):
  Q = exp(scores / eps).T
  Q /= sum(Q)
  K, B = Q.shape
  u, r, c = zeros(K), ones(K) / K, ones(B) / B
  for _ in range(niters):
    u = sum(Q, dim=1)
    Q *= (r / u).unsqueeze(1)
    Q *= (c / sum(Q, dim=0)).unsqueeze(0)
  return (Q / sum(Q, dim=0, keepdim=True)).T
```
Each iteration: scale rows to 1/K, then columns to 1/B. Final: each column renormalized to sum 1 (a per-image distribution over prototypes).

## 7. Config
| item | value | loc |
|---|---|---|
| encoder | ResNet-50, 2048-d output | App A.3 |
| projection head | 2-layer MLP -> 128-d | App A.1 |
| prototypes K | 3000 | App C.1 |
| tau | 0.1 | App A.1 |
| epsilon | 0.05 | App A.1 (also pseudocode default) |
| Sinkhorn iterations | 3 | Sec 3.1; Table 14 |
| full-res crops | 2 x 224 (or 2 x 160), RandomResizedCrop s=(0.14, 1) | App A.2 |
| small crops | V x 96, s=(0.05, 0.14) | App A.2 |
| best setting | 2x224 + 6x96 (8 crops), 800 epochs | Sec 5.1 ("800 epochs - 8 crops"); Table 4 |
| batch | 4096 over 64 V100 16GB GPUs (64 per GPU) | App A.1 |
| optimizer | LARS, lr 4.8, 10-epoch warmup, cosine to 0.0048, wd 1e-6 | App A.1 |
| prototypes frozen | first epoch | App A.1 |
| small-batch queue | 3,840 features of dim 128 (last 15 batches of 256), from epoch 15 | App A.6 |
| compute | 100 epochs of 2x224+6x96 = 6h15 on 64 V100 | Table 4 |
| linear eval | 100 epochs, lr 0.3, wd 1e-6, cosine, batch 256, on frozen 2048-d | App A.3 |

## 8. Results
- 75.3 ImageNet linear top-1, ResNet-50, 800 ep, batch 4096, 2x224+6x96 multi-crop (Fig 2 left table, row SwAV; Sec 4.1; App A.3). Supervised R50: 76.5 (same table).
- Fig 2 left comparison rows (as printed in SwAV): Colorization 39.6, Jigsaw 45.7, NPID 54.0, BigBiGAN 56.6, LA 58.8, NPID++ 59.0, MoCo 60.6, SeLa 61.5, PIRL 63.6, CPC v2 63.8, PCL 65.9, SimCLR 70.0, MoCov2 71.1. No epochs/batch printed per row.
- Fig 3 left (all 400 ep, ResNet-50, SwAV's own implementations): 2x224 / 2x160+4x96: Supervised 76.5/76.0 (200 ep), SimCLR 68.2/70.6, SeLa-v2 67.2/71.8, DeepCluster-v2 70.2/74.3, SwAV 70.1/74.1. So SwAV without multi-crop = 70.1 at 400 ep.
- Table 3 (batch 256): SimCLR 61.9 (200 ep, 2x224), MoCov2 67.5 (200 ep) and 71.1 (800 ep), SwAV 72.0 (200 ep, 2x160+4x96), 72.7 (200 ep, 2x224+6x96), 74.3 (400 ep, 2x224+6x96). Stored features: MoCov2 65,536; SwAV 3,840.
- 72.1 after 100 epochs in 6h15 (Sec 5.3).
- DeepCluster-v2 in best setting: 75.2 (Sec 5.1).
- Table 12 prototypes (400 ep, 2x160+4x96): 300->72.8, 1000->73.6, 3000->73.9, 10000->74.1, 30000->73.8, 100000->73.8.
- Table 13: learned 73.9 vs fixed random prototypes 73.1; soft 73.9 vs hard 73.3.
- Table 14 Sinkhorn iterations: 1 -> fail, 3 -> 73.9, 10 -> 73.8, 30 -> 73.7.
- Table 1 semi-sup (1% / 10% top-1): 53.9 / 70.2.

## 9. Limitations stated
- Needs enough features vs prototypes for equipartition; small batches need a queue (Sec 3.1).
- Codes from small crops degrade assignment quality and transfer, so only full-res crops make codes (Sec 3.2).
- High epsilon collapses to uniform assignment (Sec 3.1).
- One epoch of SwAV is slower in wall clock than MoCov2/SimCLR (1.2x/1.4x) (Sec 4.3).
- Later: DINO (2104.14294, Caron et al.) drops the explicit Sinkhorn and uses centering + sharpening on a teacher (`eq:center_update` in DINO approach.tex); DINOv2 replaces the teacher softmax-centering "by the Sinkhorn-Knopp (SK) batch normalization of SwAV", 3 iterations (dinov2 src/approach.tex, paragraph "Sinkhorn-Knopp centering").

## 10. Outline feedback
- The outline says "forces the codes to use all prototypes equally (Sinkhorn)". Precise wording: equal use within the batch (rows of Q sum to 1/K), soft codes, 3 iterations, no gradient through the codes.
- `viz-sinkhorn`: use eps=0.05, rows -> 1/K then columns -> 1/B, 3 iterations; Table 14 shows 1 iteration fails. Batch x prototypes in the paper is B x K with K=3000; a toy needs a small B x K (e.g. 8 x 4) and must say so.
- Say "multi-crop" is part of the 75.3; without it SwAV is 70.1 at 400 ep (Fig 3).
- Figure/table numbering trap: the 75.3 table is inside "Figure 2", not a Table.
