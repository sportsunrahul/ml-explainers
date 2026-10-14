# JEPA: a literature survey, intuition first

Reader: knows basic deep learning (what a network, a loss and gradient descent are) and nothing more.
Everything else (ViTs, masking, self-supervision, EMA teachers, world models, planning) is introduced
where it is first needed. Narrated throughout. Target ~26 min of audio in 13 chapters + interlude.

Scan date for all field counts: 2026-10-08 (notes/arxiv-scan-2026-10-08.json; 569 unique papers).

Color key (one meaning per color, whole page):
blue = context / what the model sees; yellow = target / what it must predict; green = predictor and its output;
red = loss, energy, gradient; brown = frozen or discarded (EMA target encoder, pixels thrown away).

3D figures (pe3.js, orbitable, reduced-motion safe): hero, ch-ebm energy surface, ch-collapse cloud,
ch-vjepa video cube, ch-lejepa Gaussian pull, ch-planning latent planner, ch-lewm hierarchy. All other
figures are canvas 2D or SVG. Every 3D figure says in its caption whether it is a toy or computed from data.

---

## Act I. Learning without labels

### 1. ch-ssl: Three ways to fill in a blank
Intuition to leave: you can learn from unlabeled data by hiding part of it and predicting the rest, and
WHERE you measure the prediction error (pixels vs embeddings) decides what the network must learn.
From scratch: representation = a vector that summarises an input; self-supervised = the data is its own
teacher. Three families on one picture: (a) reconstruct the hidden pixels (autoencoders, MAE),
(b) make two views of the same image land on the same embedding (SimCLR, BYOL, DINO),
(c) predict the EMBEDDING of the hidden part from the embedding of the visible part (JEPA).
Visual (2D canvas, toy): an image with a hidden block. Buttons switch the loss space: "pixels" shows the
model must reproduce leaves/texture/noise (high error everywhere); "embedding" shows a short vector of
"what is there" being predicted. A slider adds pixel noise to the target: pixel loss grows, embedding
loss does not. Prediction prompt: "which one has to memorise the exact leaves?"
Claim card (2402.11337, Balestriero & LeCun): reconstruction features are uninformative for perception;
how measured (which experiments and metric), what would falsify it.
Branch mentions: MAE 2111.06377, BYOL 2006.07733, SimSiam 2011.10566, VICReg 2105.04906, DINO 2104.14294,
data2vec 2202.03555 (predicts teacher's latent targets: the closest predecessor), MSN 2204.07141.
Extract: 2402.11337 main claim + its key figure/table; MAE's loss location (pixels, normalised); data2vec's
target (average of top-K teacher layers); BYOL's EMA update and predictor (Eq numbers).

### 2. ch-ebm: The energy view (LeCun 2022)
Intuition to leave: a model can be a function that scores compatibility, F(x, y), low when y goes with x.
Learning = shaping the energy landscape; collapse = a landscape that is flat (everything compatible).
Three architectures: generative (predict y from x in data space), joint-embedding (embed both, compare),
JEPA (embed both, PREDICT s_y from s_x with an optional latent z). Why JEPA can discard: the target
encoder may drop unpredictable detail, and the latent z absorbs the rest. H-JEPA: stack them for
multi-timescale prediction (this returns in ch-lewm).
Visual (3D, toy): an energy surface over (x, y). Drag the point; read the energy. Buttons: "collapsed"
flattens the surface; "contrastive" pushes up chosen negatives; "regularised" lowers volume by shrinking
the low-energy region. Prediction prompt: "what does the landscape look like when every y is equally good?"
Equations (exact wording from Dawid & LeCun 2306.02572, the arXiv tutorial that reprints the framework;
position paper cited by title and version only unless its PDF is supplied): F(x,y) = min_z E(x,y,z);
the JEPA energy D(s_y, Pred(s_x, z)).
Extract (2306.02572): the energy definition, the latent-variable energy, the JEPA energy and the figure
numbers; the position paper's version string and date as the tutorial cites it; the two collapse-avoidance
families (contrastive vs regularised) in their words.

## Act II. JEPA made real

### 3. ch-ijepa: I-JEPA, predicting the missing block
Intuition to leave: mask a few large blocks, encode what is visible, and predict the TARGET ENCODER's
embeddings of the hidden blocks; large blocks force semantic prediction, embeddings let it skip pixels.
From scratch: ViT = cut the image into patches, embed each, run attention; the encoder output is one
vector per patch. Then: context encoder (blue), target encoder = EMA copy (brown, no gradient),
predictor (green, narrow ViT fed mask tokens with position embeddings), L2 loss per target patch (red).
Masking: 4 target blocks (scale .15-.2, aspect .75-1.5), one context block (scale .85-1) minus targets.
Visual (2D, computed from the paper's masking config): masking playground. "Sample masks" draws new
targets/context on a 14x14 grid using the exact ranges; readout gives the counts. "Rasterised" and
"random" show the ablated alternatives. Then the predictor stage: mask tokens flow in, predictions out.
Prediction prompt: "what happens if the target blocks are tiny single patches?"
Equations: the loss (paper's Eq.), the EMA update, built term by term; each term lights its figure part.
Architecture walkthrough (ArchFlow, FULL): forward with computed shapes for ViT-B/16 and ViT-H/14 at 224
(patches, tokens, context tokens, predictor width/depth, target count), backward: which weights get a
gradient (context encoder + predictor), which are frozen-by-EMA (target encoder), what the loss grades.
Results: IN-1k linear probe for ViT-H/14; 1% low-shot; compute vs MAE and iBOT (GPU hours); ablation table
rows: predict in pixel space vs representation space; masking strategy; predictor depth.
Interlude after this chapter (not narrated): one I-JEPA training step end to end.
Extract (2301.08243v3 + repo README/config): all config values needed for shapes (patch size, widths,
depths, heads, predictor width/depth, mask ranges, batch, epochs, optimizer, schedule, EMA momentum
schedule), the loss equation with number, results tables and ablations above, GPU-hour comparison.

### 4. ch-collapse: Why doesn't it collapse?
Intuition to leave: "output the same vector for everything" solves every prediction loss with zero error;
JEPAs avoid it not by a term in the loss but by an asymmetry (EMA target, stop-gradient, a predictor),
and theory says this asymmetry prefers slow, predictable features and ignores noise.
From scratch: show the trivial solution first. Then the three asymmetries (BYOL/SimSiam lineage).
Theory, each as a claim card with a toy: 2211.10831 (JEPAs focus on slow features: with noisy fast
features and clean slow ones, the predictive loss keeps the slow ones), 2407.03475 (deep linear
self-distillation: features with high "influence" and low noise are learned first; noisy ones are not),
2210.07277 (joint-embedding methods carry a hidden uniform-cluster prior; hurts on imbalanced data).
Visual (3D, toy): an embedding cloud driven by a tiny linear JEPA trained live in the browser on synthetic
data with one slow and one noisy dimension. Toggles: EMA on/off, predictor on/off, stop-grad on/off.
Watch the cloud collapse to a point, or keep spread while the noisy axis flattens. A side chart plots the
singular values of the embeddings over steps (dimensional collapse). Prediction prompt before each toggle.
Equations: the linear-dynamics statement from 2407.03475 (their main equation) in words then symbols.
Extract: the three papers' central claims, setups and the exact statement or equation each proves; what
experiment would falsify each.

### 5. ch-vjepa: V-JEPA, adding time
Intuition to leave: a video is a 3D block of patches; mask tubes through time so the model cannot copy
from the next frame; predict their embeddings; nothing else (no pixels, no contrastive pairs).
Tubelets (2 frames x 16 x 16), 3D multi-block masks: short-range (several small blocks) and long-range
(few large blocks); the masks span all frames. Feature prediction vs pixel prediction ablation; attentive
probe for frozen evaluation; data mixture (VideoMix2M). Results on K400, SSv2, IN1k frozen.
Visual (3D, computed from the masking config): the video cube (T x H x W voxels) with tube masks you can
rotate; buttons "short-range", "long-range", "both"; readout gives masked fraction. Prediction prompt:
"why not mask random voxels?"
Delta architecture block (not a full stepper): what changes from I-JEPA: input shape, tubelet embedding,
mask spanning time, predictor outputs, loss (L1 in V-JEPA).
Extract (2404.08471v1 + repo README): tubelet size, frames, resolution, mask config numbers, loss (L1 vs
L2), results table rows above, the pixel-vs-feature ablation row, model sizes.

### 6. ch-iwm: The predictor is a world model
Intuition to leave: if you tell the predictor WHAT transformation separates context and target (the
"action"), it learns to simulate that transformation in embedding space: a world model. Equivariant vs
invariant representations; a weak predictor forces invariance, a strong conditioned predictor allows
equivariance; the predictor can then be fine-tuned for downstream tasks cheaply.
Visual (2D canvas, toy): a latent plane; an input point; action buttons (blur, color shift, crop); the
predictor arrow moves the point. Toggle "conditioned predictor" vs "blind predictor": blind one must map
all actions to the same place (invariance). Prediction prompt: "what must the blind predictor do?"
Claim card for IWM results (equivariance measured how; predictor fine-tuning results).
Extract (2403.00504v1): the conditioning mechanism, the equivariance metric, key table rows, how the
predictor is fine-tuned, the capacity finding.

### 7. ch-vjepa2: V-JEPA 2, scale, then act
Intuition to leave: scale the self-supervised video model four ways, then freeze it and bolt on a small
ACTION-CONDITIONED predictor trained on a little robot video; plan by searching actions whose predicted
embedding lands near the goal image's embedding. Zero-shot on a real arm.
Part A (scale): data (hours, clips), model (ViT-g), progressive resolution and clip length, schedule;
results on SSv2, Diving48, EK100 anticipation, video QA after LLM alignment.
Part B (act): V-JEPA 2-AC: frozen encoder, action-conditioned predictor (block-causal attention), trained
teacher-forced + rollout on Droid hours; planning = CEM over action sequences minimising L1 to the goal
embedding; results: reach, grasp, pick-and-place success rates in two labs.
Claim card (2502.11831): intuitive physics via surprise = prediction error; IntPhys / GRASP / InfLevel
accuracy vs pixel-prediction and VLM baselines; what would falsify it.
Visual A (2D): the four scaling ingredients as a stacked bar from the paper's ablation (sourced).
Visual B (3D, toy on top of the real algorithm): latent planner. Current embedding, goal embedding, CEM
samples as faint candidate paths, the chosen plan solid; "replan" button; a slider for horizon.
Prediction prompt: "what does the planner minimise?"
Architecture walkthrough (ArchFlow, FULL) for 2-AC: frames -> tokens -> frozen encoder (brown, gradient
stops: nothing trainable upstream) -> action + proprio tokens -> predictor (green, trainable) -> L1 vs
teacher embeddings (red); backward shows only the predictor's parameters updating.
Extract (2506.09985v1 + repo README): the four ingredients with their ablation numbers, data hours and
clip counts, model sizes, resolution/frames, 2-AC predictor size and attention pattern, Droid hours,
loss, CEM settings (samples, iterations, horizon), robot results table, intuitive-physics numbers from
2502.11831 (datasets, accuracies, baselines).

### 8. ch-vjepa21: V-JEPA 2.1, dense features
Intuition to leave: let EVERY token carry loss, not only the masked ones, and supervise intermediate
layers too; the features become spatially precise without losing the global picture.
Dense predictive loss; deep self-supervision; multimodal tokenizers (images and videos in one model);
sizes (ViT-g/G, 1B/2B); gains on dense tasks and robotics (grasp success vs 2-AC).
Visual (2D): before/after token grid: which tokens receive loss in V-JEPA 2 vs 2.1; a layer stack showing
where losses attach. Buttons "V-JEPA 2" / "V-JEPA 2.1".
Extract (2603.14482v3): the dense loss definition and equation, which layers get deep supervision, the
tokenizer change, model sizes, the headline dense-task numbers and the robotics delta.

## Act III. Cleaning up the recipe

### 9. ch-lejepa: LeJEPA, one regulariser instead of the heuristics
Intuition to leave: instead of EMA + stop-gradient + predictor tricks, add ONE term that pushes the
embedding distribution toward an isotropic Gaussian; theory says that distribution minimises downstream
risk; it is tested cheaply by projecting onto random 1D directions (sketching) and comparing with a
Gaussian. One hyperparameter, stable across architectures and scales.
SIGReg (sketched isotropic Gaussian regularisation): random directions, a 1D goodness-of-fit statistic
(Epps-Pulley), averaged. Loss = prediction + lambda * SIGReg. Results: IN-1k linear for ViT-H/14, many
architectures/scales, in-domain pretraining on small datasets vs frozen DINOv2/v3 transfer.
Companions: 2510.05949 (the anti-collapse term estimates data density: JEPA-SCORE), 2605.26379 (LeJEPA
linearly identifies the world's latents; Gaussian is the unique distribution for which this holds).
Visual (3D, toy + computed statistic): an embedding cloud (anisotropic, clumpy) and a direction you can
drag on the sphere; a side panel shows the 1D projection histogram vs the Gaussian CDF and the Epps-Pulley
statistic computed live. Button "apply SIGReg" eases the cloud to isotropic; the statistic drops.
Prediction prompt: "which direction will the test fail on?"
Equations: SIGReg as written (their equation number), the full LeJEPA loss, built term by term.
Light architecture block: the "what is removed" table (EMA, stop-grad, predictor, prototypes, etc.) and
what stays, from the paper's own comparison.
Extract (2511.08544v3 + repo README): SIGReg equation and the Epps-Pulley form, number of directions,
lambda default, the optimality theorem statement, results tables rows above, the list of removed
heuristics, parameter range covered (to 1.8B), the "~50 lines" claim if present; from 2510.05949 the
density statement and equation; from 2605.26379 the main theorem in one sentence.

## Act IV. World models you can plan in

### 10. ch-planning: Planning in latent space
Intuition to leave: a world model predicts the next embedding given the current one and an action; with
one, you can plan by imagining action sequences and keeping the one whose imagined future is closest to
the goal; no reward, no policy training.
From scratch: state, action, dynamics; model predictive control in two sentences.
Three designs: DINO-WM (frozen DINOv2 patch features + ViT predictor conditioned on action and proprio;
CEM/gradient planning; zero-shot goal reaching on PushT, point maze, etc.), PLDM 2502.14819 (JEPA latent
dynamics from reward-free offline data; vs model-free RL under distribution shift; "a case for planning"),
Navigation World Models 2412.03572 (conditional diffusion transformer predicting future egocentric views
from actions; plan by sampling trajectories; scale).
Visual (3D, toy on the real algorithm): the latent planner, full version: a floor grid of states, goal
marker you can drag, CEM iterations button (watch the candidate fan narrow), horizon slider, readout of
the best imagined distance. "Frozen features" vs "end-to-end" labels for the next chapter.
Claim cards for the three papers' headline results.
Extract (2411.04983v2, 2502.14819v4, 2412.03572v2 + READMEs): predictor sizes, planning algorithm settings,
headline success rates and the comparison baselines, data regimes.

### 11. ch-lewm: End to end from pixels, and up the hierarchy
Intuition to leave: with SIGReg you no longer need a pretrained encoder or EMA; LeWorldModel trains encoder
and predictor from raw pixels with two loss terms and plans fast; H-JEPA stacks such models so higher
levels predict farther ahead in coarser latents and hand subgoals down.
LeWorldModel 2603.19312v3: two terms (next-embedding prediction + SIGReg), parameter count (~15M), training
cost, stability without EMA/pretrained encoders, planning speed; results vs DINO-WM-style baselines.
H-JEPA 2610.06805v1: hierarchy of action-conditioned JEPAs, each level its own latent and horizon; top-down
planning; when factors evolve at separated timescales higher levels drop fast detail.
Visual (3D, toy): stacked latent planes; a long trajectory on the top plane becomes several short ones on
the plane below; "plan top-down" button animates subgoals descending. Prediction prompt: "what does the
top level forget on purpose?"
Equations: LeWM's two-term loss, built term by term.
Light architecture block for LeWM (what is trainable: everything; what the loss grades).
Extract (2603.19312v3 + README, 2610.06805v1): the loss, sizes, training time and hardware, planning time,
headline results; H-JEPA's level definitions, horizons, planning procedure, headline results.

## Act V. The field

### 12. ch-family: One idea, every modality
Intuition to leave: the recipe is modality-agnostic: define context and target, encode both, predict in
embedding space; every branch changes only what the blocks are.
Branches (one or two sentences each, placed on the tree): MC-JEPA 2307.12698 (motion + content), A-JEPA
2311.15830 (audio spectrograms), MJEPA 2606.25225 (audio-visual, 2026), Point-JEPA 2404.16432 and 3D-JEPA
2409.15803 (point clouds), Graph-JEPA 2309.16014, LLM-JEPA 2509.14252 (text views), VL-JEPA 2512.10942
(predict text embeddings from video), D-JEPA 2410.03755 (generation), Causal-JEPA 2602.11389 (object-level
masking), EB-JEPA library 2602.03604, Var-JEPA 2603.20111 (variational view).
Visual (SVG): an interactive family tree; click a node for its delta card (context / target / predictor /
what changed).
Extract (branch tier, abstracts + one key table each): the one-sentence delta and one headline number.

### 13. ch-map: The 2026 explosion
Intuition to leave: the field went from a position paper to hundreds of papers in four years, and most of
the 2026 growth is world models and planning; the live debates are about what a deterministic latent
predictor can and cannot represent.
Visual (SVG, computed from the scan): papers per month 2023-2026 and a domain breakdown; the method
(queries, date, title-keyword classification) stated on the figure. Open problems with one paper each:
stochastic futures (MoP-JEPA 2607.05238), velocity blind spot 2610.04585, physical grounding 2608.06799,
planning metric mismatch 2607.25337, training dynamics 2610.02344, probabilistic views (Var-JEPA, VJEPA
2601.14354).
Claim card: how the scan was done and what it misses (title/abstract only, arXiv only).
Extract: none beyond the scan JSON and the listed abstracts.

### ch-timeline: 2020-2026 on one line (SVG; sourced dates from abs pages)
### ch-summary: one paragraph
### ch-deeper: where to go deeper, with paper, section and equation per item

---

## Full ArchFlow steppers: ch-ijepa (+ interlude), ch-vjepa2 (2-AC). Light blocks: ch-vjepa, ch-lejepa, ch-lewm.
## Claim cards: ch-ssl (2402.11337), ch-collapse (three theory papers), ch-iwm, ch-vjepa2 (intuitive physics),
## ch-planning (three), ch-map (scan).
