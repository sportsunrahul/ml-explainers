# Diffusion models, intuition first: outline

Reader: no assumed background. Wants history, math and practice. All 15 papers kept. Page only
for now (no narration), but every viz keeps the cue contract so narration can be added later.

Deviation from the skill default (6-9 chapters): 12 content chapters in 5 acts, because the
reader asked to "explain everything" across 15 papers. DDPM gets the full arc (ch 2-4); every
later paper is a delta chapter against its predecessor.

Color key (whole page):
- yellow = real data (x_0, training samples)
- brown  = pure noise (x_T, the true noise epsilon)
- blue   = what the model predicts (predicted noise, score arrows, velocity field, x0-hat)
- green  = generated samples and their trajectories
- red    = error / loss

Shared toy: a 2D "spiral" dataset of a few hundred points. On a finite dataset the ideal
denoiser E[x_0 | x_t] has a closed form (a softmax-weighted average of the data points), so
most chapters show the EXACT optimal model on the toy instead of a fake one. Badges: "Computed"
when a real formula from a paper runs on the toy; "Toy model" when illustrative; "Sourced"
when numbers come from a paper table.

## Act I: the idea

### 1. ch-generate: What does it mean to generate?
Intuition: an image is one point in a huge space; "the data" is a cloud; generating = drawing a
new point from that cloud. Turning structure into noise is easy; diffusion learns to undo it in
many small steps. Also: why GANs (unstable, mode collapse), VAEs (blurry) and flows (restricted
architectures) struggled, and Sohl-Dickstein 2015's ink-in-water idea.
Viz viz-dissolve (Computed): the spiral dissolves into a Gaussian blob under a noise slider;
"run it backwards" plays the reverse with the ideal denoiser as a teaser. Prediction prompt:
"what shape will the cloud become?"
Papers: Sohl-Dickstein (origin).

## Act II: DDPM, the full arc

### 2. ch-forward: The forward process, adding noise on a schedule
Intuition: each step shrinks the point a little toward the origin and adds a little noise; after
enough steps nothing of the data survives, and you can jump to any step in one shot.
Viz viz-schedule (Computed): one data point, its x_t distribution as t slides; side plot of
alpha_bar_t for DDPM's linear schedule vs Improved DDPM's cosine schedule (real formulas,
T = 1000); shows the linear schedule wasting its last steps on near-pure noise.
Math: q(x_t|x_{t-1}), closed form q(x_t|x_0), alpha_bar.
Papers: DDPM, Improved DDPM (cosine schedule).

### 3. ch-denoise: Learning to undo it: predict the noise
Intuition: train a network to look at a noisy point and guess the noise that was added; that
guess is all you need to step backwards. The ELBO, gently: the fancy variational bound reduces
to "mean-squared error on the noise".
Viz viz-predict-noise (Computed): drag a noisy point; see the true noise (brown), the ideal
model's prediction (blue, exact on the toy), the error (red); a noise-level slider shows why
the task is ambiguous at high noise (the prediction points at the data's average).
Math: L_simple, the posterior mean, how epsilon relates to x0-hat.
Papers: DDPM, Improved DDPM (learned variance, hybrid loss, briefly).

### 4. ch-sample: Sampling: walking back from noise
Intuition: start from pure noise, repeatedly remove predicted noise and add back a bit of fresh
noise; a cloud of particles condenses into the data. It works, but takes ~1000 network calls.
Viz viz-reverse (Computed): particles from a Gaussian, scrub the reverse steps; trails in green;
step-count selector shows quality collapsing when you naively take fewer steps.
Math: DDPM Algorithm 2 update.
Results (Sourced): DDPM CIFAR-10 FID / IS (Table 1); a sourced plot from DDPM's own plot_data CSVs of how
quality and distortion evolve along the reverse chain (coarse structure first, detail last).
Improved DDPM's few-step results are figure-only, so only its textual claims appear.
Papers: DDPM, Improved DDPM.

## Act III: the score view and faster sampling

### 5. ch-score: Scores, Langevin, and the SDE that unifies everything
Intuition: the score is an arrow field pointing toward more likely data; predicting noise and
predicting the score are the same job up to a scale. NCSN learned scores at many noise levels
and sampled with annealed Langevin dynamics. Score SDE showed DDPM and NCSN are two
discretizations of one continuous process, with a deterministic twin (the probability-flow
ODE) that gets the same marginals.
Viz viz-score-field (Computed): the exact score field of the noised toy at a noise-level slider;
buttons: annealed Langevin particles; SDE vs ODE trajectories from the same start points.
Math: score definition, epsilon = -sigma * score, reverse SDE, probability flow ODE.
Papers: NCSN, Score SDE.

### 6. ch-fast: Fewer steps: DDIM and EDM
Intuition: if the path is deterministic you can take big strides along it; the skill is choosing
where to step and how to step (step sizes concentrated at low noise, second-order Heun).
Viz viz-steps (Computed): same start noise; DDPM ancestral vs DDIM (eta = 0) vs EDM Heun at a
steps slider (e.g. 5 to 100); endpoints vs the data; error readout.
Math: DDIM update (x0-hat + direction + noise, eta), EDM sigma schedule with rho, Heun step,
preconditioning (brief).
Results (Sourced): DDIM FID vs steps table; EDM CIFAR-10 FID 1.97 unconditional / 1.79 conditional,
both at 35 network evaluations. Notation warning box: DDIM's alpha_t is DDPM's alpha_bar_t.
Papers: DDIM, EDM.

## Act IV: control and scale

### 7. ch-guidance: Steering: classifier guidance and classifier-free guidance
Intuition: add a push toward "looks like class y"; turning the push up trades diversity for
fidelity. CFG gets the push for free by training one model with and without the label.
Viz viz-guidance (Computed): two-class toy (spiral arm A vs B); guidance weight slider; samples
pile onto the class's most typical region; diversity readout.
Math: classifier-guided mean shift; CFG epsilon combination.
Results (Sourced): ADM FID vs BigGAN (point results; ADM's scale sweep is figure-only); CFG's w sweep
tables (FID bottoms out near w = 0.3, then rises while IS keeps climbing: the tradeoff made visible).
Papers: Beat GANs (ADM, architecture + classifier guidance), CFG.

### 8. ch-latent: Diffuse in a compressed space: latent diffusion
Intuition: most pixels are perceptual detail an autoencoder can restore; do the expensive
diffusion on a small latent grid, and inject text via cross-attention.
Viz viz-latent (Toy + Sourced): a downsampling-factor selector shows the pixel grid vs latent
grid and the relative cost, with LDM's per-f reconstruction quality (Table 8); the f-sweep curves
are figure-only, so the chapter cites the text's summary. Latent shape: the paper's text-to-image
model is 32x32x4 at 256 px (64x64x4 at 512 is Stable Diffusion, not this paper).
Math: LDM objective in latent space; cross-attention Q/K/V.
Papers: LDM (Stable Diffusion).

### 9. ch-dit: Swap the U-Net for a transformer
Intuition: cut the latent into patches, treat them as tokens; quality tracks compute (Gflops).
Viz viz-dit (Sourced): patch-size selector -> token count and Gflops; scatter of FID vs Gflops
from the paper.
Math: token count, adaLN-Zero modulation.
Papers: DiT.

## Act V: straight paths and one step

### 10. ch-flow: Rectified flow and flow matching: learn straight paths
Intuition: pair noise with data, draw straight lines, learn the velocity along them. Lines
cross, so the learned flow bends; "reflow" re-pairs and straightens it; straighter paths need
fewer steps. Flow matching gives the general recipe (conditional paths, OT path).
Viz viz-reflow (Computed): random pairs as lines; learned (exact marginal) velocity field
trajectories; reflow button; Euler step selector with endpoint error.
Math: x_t = t x_1 + (1-t) x_0 interpolation, velocity regression loss, conditional FM loss.
Papers: Rectified Flow, Flow Matching.

### 11. ch-consistency: Consistency models: jump to the end
Intuition: every point on one ODE trajectory should map to the same endpoint; train a network
to do that jump directly, giving one-step (or few-step) generation.
Viz viz-consistency (Computed): trajectories; pick any point, see its one-step jump; two-step
refinement.
Math: self-consistency property, boundary condition parametrization, CD loss.
Papers: Consistency Models.

### 12. ch-sd3: Where it landed: SD3
Intuition: rectified flow + a transformer with separate text and image streams, trained with
more weight on middle noise levels, scales predictably.
Viz viz-timesteps (Computed + Sourced): logit-normal timestep density with m/s sliders and the
resolution shift; sourced scaling trend.
Papers: SD3 (and how it inherits LDM, DiT, rectified flow, CFG).
Note: SD3 runs time the other way (t = 0 data, t = 1 noise); the page keeps one convention and
flags the flip. Human-preference win rates exist only as bar charts, so they are not quoted.

## Closing
- ch-timeline: family tree 2015 -> 2024 with arrows "inherits from".
- ch-summary: one paragraph.
- ch-deeper: per paper, which section / equation to read next.
