# IWM, Image World Models (notes for ch-iwm)

Lens: architecture + analysis. Sources: sources/iwm/{abs.html, paper.pdf, src/paper.tex}. No repo (meta.json
repo: null; none in the LaTeX). Table/figure numbers and pages verified against paper.pdf. Appendix tables and
figures are numbered S1..; appendix sections print without letters in the PDF and are cited by title + page.

## 1. Identity
- Title: Learning and Leveraging World Models in Visual Representation Learning
- arXiv: 2403.00504v1 (version read). abs.html dateline: "[Submitted on 1 Mar 2024]" -> published 2024-03-01.
  abs.html comments: "23 pages, 16 figures".
- Authors: Quentin Garrido (first) ... Yann LeCun (last); 6 authors.
- Venue: not stated in abs.html or the LaTeX.
- Repo: none given.
- PDF: 23 pages. Main text pp. 1-10, appendix pp. 13-23.

## 2. Core problem
JEPA predictors are world models (they predict the effect of an "action", the masking, in latent space), but
representation learning discards the predictor after pretraining. The paper asks (a) what it takes to learn a
predictor that can apply a broader set of transformations (photometric augmentations) in latent space, and
(b) whether that predictor is worth keeping for downstream tasks.

## 3. What prior methods did and why that hurt (per the paper)
- Contrastive / invariance methods (SimCLR, MoCo, BYOL, SimSiam): "If one does not condition the predictor on
  the transformation parameters, then the best we can hope for is learning representations that are invariant
  to the data transformations" (Sec 1). Invariance removes information (e.g. color) from the encoder.
- MAE / I-JEPA: the decoder or predictor is a world model for the masking action only, and "the learned world
  model in self-supervised learning is typically discarded after pretraining" (Sec 1). "We postulate that
  discarding the world model in representation learning is wasteful."
- Equivariant SSL methods need an extra invariance loss to keep representation quality; a JEPA does not
  because semantics come from latent inpainting (Sec 3).

## 4. The one key idea
Tell the predictor which transformation separates context and target (color-jitter deltas plus flags for
grayscale, blur, solarization, in addition to the masked positions). With enough capacity and hard enough
transformations, the predictor learns to apply the transformation in embedding space (an equivariant world
model), the encoder keeps more information, and the predictor can be fine-tuned as a cheap task head.

## 5. Mechanism, step by step
From Sec 3 (pp. 3-4) and Appendix "Experimental details" (p13, label app:exp-details), Fig S1 (p13).
1. From image I: random crop (scale 0.3-1) and horizontal flip (p 0.5) -> I'. Both views share this
   geometric step; crop and flip are NOT part of the action.
2. Target y: I' + color jitter (p 0.8; brightness 0.4, contrast 0.4, hue 0.1, saturation 0.2). No destructive
   augmentation on the target, "to ensure that the target has as much information as possible".
3. Source x: I' + another color jitter (same ranges) + destructive augmentations: Gaussian blur (radius
   0.1-2, p 0.2), solarization (p 0.2 in the appendix text; the default row of Table S2 shows 0.1), grayscale
   (p 0.2). Then mask M_x = union of 4 rectangular blocks, each of area 0.15-0.2 of the image, aspect ratio
   0.75-1.5; "All of the patches in M_x are then dropped from the source x" (p13).
4. Action a_{x->y}: "the transformation parameters associated with the transformation of x to y, i.e., the
   invert of the initial transformation process. a_{x->y} contains information about the color jitter
   difference between x and y as well as information on whether or not each destructive augmentation was
   applied" (Sec 3). A vector a in R^k of k scalars (p13).
5. Encode: z_x = f_theta(x) (ViT-B/16, visible patches only); z_y = f_theta^EMA(y) (EMA encoder, full target).
   "The use of the EMA network is crucial to avoid collapsed solutions."
6. Conditioning (default = feature conditioning, p13): for each target position, take a mask token m in R^d,
   add its positional embedding, concatenate a in R^k, pass through a 3-layer fully connected network with ReLU
   and widths d, d, d -> a mask token m_a that carries both where to predict and what transformation to apply.
   Alternative (sequence conditioning, Sec 4.2): append transformation tokens to the predictor input, each
   through its own linear layer to break permutation symmetry. Table 1 (p5): MRR none 0.00, sequence 0.82,
   feature 0.79; feature is used "as it leads to higher downstream performance".
7. Predictor p_phi (ViT blocks; 12 or 18 layers, width 384) takes the embedded source patches, a_{x->y} and
   the mask tokens m_a and outputs \hat z_y = p_phi(z_x, a_{x->y}, m_a).
8. Loss: squared L2 between \hat z_y and z_y at the predicted positions (equation below). Gradients to theta
   and phi; the EMA encoder gets none.
9. Nomenclature: IWM_{X,Y}^Z, X = predictor depth, Y = predictor embedding dim, Z = Inv or Equi "depending on
   the capabilities of the world model" (Sec 3.1). Main models: IWM_{12,384}^Inv and IWM_{18,384}^Equi.

Invariant vs equivariant, in their words (Sec 3, p3): "We consider a world model to be capable if it can apply
transformations in latent space, and thus learns equivariant representations. As such, we call a capable world
model equivariant [footnote: This is an abuse of language as not all considered transformations form a group,
but it is used for clarity.] and a poor world model invariant." Sec 1: "a predictor that can apply
transformations effectively is equivariant, whereas a predictor that cannot is invariant." Appendix "On the
meaning and role of invariance" (p18) defines invariance as f_theta(x) = f_theta(T(a, x)) for all a.

## 6. Equations
(a) Loss. Unnumbered display (`equation*`), Sec 3 "Loss" paragraph (p4):
```
L(x,y) = \sum_{i\in M_x^C}\| p_\phi\left(f_\theta(x),a_{x\rightarrow y},m_a \right)_i
- f_\theta^\text{EMA}(y)_i \|_2^2.
```
Paper: "The loss function used is a squared L2 distance between the predictions \hat z_y and their targets z_y".
Gloss: f_theta = encoder on the source x; f_theta^EMA = EMA encoder on the target y; a_{x->y} = action
vector; m_a = conditioned mask tokens; subscript i = patch index; sum over the predicted positions.
Notation caveat: the source defines M_x as the dropped patches ("All of the patches in M_x are then dropped
from the source x", p13) but sums the loss over M_x^C and says the mask tokens "correspond to the positions in
M_x^C" (Sec 3). Fig S1's caption says "target positions are sampled as the masked patches of the source
image". Our reading: the loss grades the masked (dropped) positions; the M_x^C superscript is a notational
slip in the .tex. Quote verbatim, do not "fix" it on the page.

(b) Marginalised invariant representation, unnumbered, Appendix "On the meaning and role of invariance"
(p18):
```
z_x^{\text{Inv}} = \frac{1}{N} \sum_{i=1}^N p_\phi\left( f_\theta(x),A_i,m_{A_i} \right)
```
Gloss: average of N predictions under randomly sampled actions A_i. Table S6 (p19): linear top-1 64.5 (N=1)
vs 64.3 / 64.6 / 64.6 / 64.4 for N = 8 / 16 / 32 / 128; no gain.

## 7. Measurement setup and config
Pretraining (Appendix "Experimental details", p13): encoder ViT-B/16, 300 epochs on ImageNet-1k; AdamW, lr
1e-3, beta1 0.9, beta2 0.999, linear warmup 40 epochs then cosine; iteration-per-epoch scale 1.25 (schedule
stretched so training ends before lr reaches 0); weight decay cosine 0.04 -> 0.4. Predictor: 12 or 18 layers,
dim 384 (ViT-L study: 18,384 Inv and 36,512 Equi; "a ratio of predictor weights/encoder weights of around 0.3
is suitable", p17). Not stated: pretraining batch size, GPU type, GPU-hours.

Equivariance metric, MRR (Sec 4.1, p4-5): "we generate a bank of augmented target images (256 in practice). We
feed the representation of the clean image through the predictor with the goal of predicting the target image.
We then compute the distance between the prediction and the augmented representation bank from which we get
the rank of the target in this NN-graph. Averaging the reciprocal ranks over multiple images and
transformations gives us MRR ... A MRR close to 1 means that the world model is able to apply the
transformation, on the contrary a MRR close to 0 means that it cannot." Fig 1 (p1) and Figs S5-S11 show the
nearest neighbour of the prediction in the same 256-image bank.

Three factors for a strong IWM (Sec 4.2, p5): conditioning, transformation complexity/strength, predictor
capacity. "We show that not caring properly for either of those leads to invariant representations."

Evaluation protocols (Appendix p13-14), all on ImageNet-1k, ViT-B/16 encoder:
- Linear: 90 epochs, average-pooled features, LARS, batch 16,384, lr 6.4.
- Attentive: 90 epochs, cross-attention head (Chen et al. 2023), AdamW lr 1e-4, batch 1024, wd 0.01.
- Encoder finetuning: 100 epochs, linear head, RandAugment rand-m9-mstd0.5-inc1, CutMix, MixUp, AdamW lr
  2e-3, wd 0.005, batch 1024, drop path 0.2, layer decay 0.65.
- Predictor finetuning (p14): "we use an attentive head on top of the predictor output. We plug the predictor
  on top of the teacher network and it is tasked with predicting the whole target image, with null
  transformation parameters." Same augmentations as encoder finetuning; 100 epochs; batch 1024; AdamW lr
  1e-3, 5-epoch warmup, cosine; wd 0.1; no layer decay; drop path 0.2 in the predictor; "if the predictor is
  pretrained we divide it's learning rate by 10". The encoder stays frozen.
- End-to-end: predictor-finetuning recipe, encoder lr also /10, layer decay 0.9 over encoder+predictor,
  lr 2e-3.
- Multitask predictor tuning (Sec 5.2, Fig S2): one learned task token per task + task-specific head; batch
  split evenly across tasks; 512 per task; 50 ImageNet-epoch equivalents.
- Segmentation (p14): ADE20k, UperNet head, 160k iterations, mIoU; last 4 layers of predictor (or encoder)
  concatenated.

## 8. Results worth remembering
Capacity and transformation strength (Table 2, p5; MRR; columns = predictor (depth, dim)):
| Transformations | I-JEPA (12,384) | IWM (12,384) | IWM (18,384) |
|---|---|---|---|
| Jitter | 0.00 | 0.11 | 0.25 |
| + Destructive | 0.00 | 0.09 | 0.79 (default) |
| + Strong Jitter | 0.00 | 0.81 | 0.85 |
Text (Sec 4.2, p5): "For 12 layers, jitter equivariance is achieved 1 out of 5 times whereas for the 18 layers,
it is achieved 4 out of 5 times." Full sweep in Table S2 (p16): 12-layer default MRR 0.09, linear 74.5,
attentive 77.0, pred-ft 81.3; 18-layer default MRR 0.79, linear 67.5, attentive 75.1, pred-ft 83.3.

Conditioning (Table 1, p5): MRR none 0.00, sequence 0.82, feature 0.79.

Predictor finetuning, ImageNet-1k top-1 (Table 4, p7; ViT-B/16; "Pretrained" column shows gain over the
random-init predictor):
| Method | Epochs | Encoder ft (no predictor) | Frozen enc + random-init predictor | Frozen enc + pretrained predictor | End to end |
|---|---|---|---|---|---|
| MAE | 300 | 82.7 | 82.4 | 82.7 (+0.3) | 82.3 |
| MAE | 1600 | 83.6 | 83.0 | 83.1 (+0.1) | 83.3 |
| I-JEPA | 300 | 83.0 | 79.1 | 80.0 (+0.9) | 82.0 |
| IWM_{12,384}^Inv | 300 | 83.3 | 80.5 | 81.3 (+0.8) | 82.7 |
| IWM_{18,384}^Equi | 300 | 82.9 | 81.5 | 83.3 (+1.8) | 84.4 |
Text (Sec 5.1, p7): "when looking at IWM with an equivariant world model, we see a gain of 1.8 points over a
random predictor"; predictor finetuning of the equivariant model (83.3) matches encoder finetuning of the
invariant model (83.3). Table S1 (p15) adds IWM_{12,384}^Equi: 82.7 / 81.3 / 82.7 (+1.4) / 83.3, and
iBOT 1600: 84.0 / 83.0 / N/A / 82.8.

Prediction-task ablation for predictor finetuning (Table 3, p6): Default 82.9; + Teacher 83.2 (+0.3); + Null
latents 83.3 (+0.1, default); + Pred only one token 82.8 (-0.5).

Peak from one pretraining (Table 5, p7; frozen encoder / any protocol): IWM_{18,384}^Equi 83.3 / 84.4;
IWM_{12,384}^Inv 81.3 / 83.3; iBOT (1600 ep) 83.0 / 84.0; MAE (1600) 83.1 / 83.6; I-JEPA 80.0 / 82.0;
DINO 82.0 / 82.8; MOCOv3 76.4 / 83.2.

ADE20k mIoU (Table 6, p7; encoder ft / predictor ft / end to end): I-JEPA 44.2 / 45.4 / 45.1; IWM_{12,384}^Inv
45.6 / 45.7 / 46.5; IWM_{18,384}^Equi 44.2 / 46.8 / 47.0.

Efficiency (Fig 3, p6; Sec 5.1): at comparable finetuned parameter counts, predictor finetuning beats encoder
finetuning "by around 1 point compared to MAE, and by 1.5 points over IWM". ViT-L (Table S4, p17): I-JEPA
encoder 84.1 / predictor 79.9; IWM_{18,384}^Inv 84.3 / 81.5; IWM_{36,512}^Equi 83.7 / 85.0 / end-to-end 85.4.
Sec 5.1: ViT-B end-to-end 84.4% vs ViT-L encoder ft 84.3% "with a fraction of the parameters (121M vs 307 M)".

Multitask (Table 7, p8; IWM_{18,384}^Equi; single-task / multi-task): ImageNet 80.8 / 79.6 (-1.2); iNat18
72.4 / 72.0 (-0.4); SUN397 75.6 / 78.2 (+2.6); Places205 64.8 / 64.1 (-0.7); average 73.4 / 73.5 (+0.1).

Linear vs attentive (Table 8, p8; ImageNet-1k): MoCoV3 76.3 / 76.4; MAE 300 ep 60.2 / 73.5; MAE 1600 ep
68.0 / 76.0; I-JEPA 70.0 / 75.0; IWM_{12,384}^Inv 74.5 / 77.0; IWM_{18,384}^Equi 67.5 / 75.1. Text (Sec 6):
the invariant IWM behaves like contrastive methods (strong linear), the equivariant one like MIM (weaker
linear, stronger with bigger heads). Fig 4 (p9): equivariance (MRR) correlates with linear and predictor-ft
performance, "hardly correlated to Attentive probing". Fig 5 (p9): abstraction spectrum, contrastive at one
end, MIM at the other, IWM spans it.

OOD attentive probing (Table S5, p17): IWM_{18,384}^Equi iNat18 54.2 vs Inv 51.6 (+2.6); SUN397 71.7 vs 71.0;
Places205 60.5 vs 59.4.

What would falsify the main claims: a conditioned, deep predictor with MRR near 0 (Table 2 pattern breaks); or a
random-init predictor matching the pretrained one on IWM_Equi (the +1.8 in Table 4 vanishes); or an invariant
model beating the equivariant one in predictor finetuning (Tables 4, 6, S4 all show the opposite).

## 9. Limitations the paper states
- "A limitation of world models learned on images is that the task they solve is not aligned with most
  downstream tasks" (Sec 5, p6); hence the predictor must be finetuned, not used as is.
- Equivariant IWM loses in linear probing (67.5 vs 74.5) and in encoder finetuning (82.9 vs 83.3); the trade is
  abstraction vs peak performance (Sec 6).
- Grayscale is not properly invertible; the world model shows "some inaccuracies when inverting grayscale"
  (Sec 4.3).
- Predictor finetuning adds parameters at inference (Sec 5.1); multitask tuning amortises it.
- Scaling the predictor width "can lead to instabilities" (p17).
No later paper among my assigned sources addresses these.

## 10. Outline feedback (ch-iwm)
1. The toy's action buttons list "crop". Crop and flip are shared pre-transforms applied to both views and are
   NOT in the action vector. The parametrised actions are the color-jitter difference (brightness, contrast,
   saturation, hue) plus binary flags for grayscale, blur and solarization. Replace "crop" with "grayscale"
   or "solarize".
2. The intuition names two factors (conditioning, capacity). The paper names three: conditioning,
   transformation strength, capacity (Table 2 rows are the strength axis; the +Strong Jitter row makes even the
   12-layer predictor equivariant, 0.81). Add strength or the claim card misstates the finding.
3. "a weak predictor forces invariance, a strong conditioned predictor allows equivariance" matches the paper,
   but the paper's mechanism for "blind predictor must map all actions to the same place" is stated in Sec 1
   (no conditioning -> at best invariant representations) and measured as MRR 0.00 (Table 1). Cite those two.
4. Claim card: the equivariance metric is MRR against a bank of 256 augmented targets (Sec 4.1); headline
   numbers are Table 2 (0.09 -> 0.79 going 12 -> 18 layers at default augmentations) and Table 4 (83.3 vs
   81.5 random-init, +1.8; end-to-end 84.4).
5. The loss is squared L2 (not L1 as in V-JEPA) and the predictor is conditioned on both positions and the
   action; this is the natural contrast with ch-vjepa's z = positions only.
6. The outline says "the predictor can then be fine-tuned for downstream tasks cheaply": the paper's own
   efficiency claim is in finetuned-parameter terms (Fig 3) and the 121M vs 307M comparison; it also notes that
   inference-time parameters go UP. Phrase as "cheaper to adapt", not "cheaper to run".
