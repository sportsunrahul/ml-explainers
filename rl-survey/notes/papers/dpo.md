# DPO: Direct Preference Optimization (Rafailov et al. 2023)

## 1. Identity
- Title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv 2305.18290, version
  read v3 (2024-07-29); v1 2023-05-29. NeurIPS 2023 (PDF footer). PDF 27 pages.
- Authors: Rafael Rafailov (first), Chelsea Finn (last; 6 authors; first three equal contribution).
- Sections: 1 Introduction, 2 Related Work, 3 Preliminaries, 4 Direct Preference Optimization, 5 Theoretical
  Analysis of DPO (5.1 Your Language Model Is Secretly a Reward Model, 5.2 Instability of Actor-Critic Algorithms),
  6 Experiments (6.1-6.4), 7 Discussion; App. A derivations, App. B DPO Implementation Details and Hyperparameters.
- Source note: the LaTeX defines `\rev{old}{new}` as `{#2}`, so only the second argument prints. v3 reworded several
  claims through this macro (see Sec. 10).

## 2. Core problem
RLHF fits a reward model and then runs RL (PPO) to maximize it under a KL constraint. The pipeline is complex,
unstable, and samples from the policy in the training loop.

## 3. What prior methods did and why that hurt (per the source)
"RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences,
and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without
drifting too far from the original model." (abstract). The RLHF pipeline involves "training multiple LMs and sampling
from the LM policy in the loop of training, incurring significant computational costs" (Sec. 1, p2).

## 4. Key idea
The KL-constrained objective (Eq. 3) has a closed-form optimal policy (Eq. 4). Invert it to write the reward in terms
of the policy (Eq. 5); in the Bradley-Terry likelihood the partition function cancels, so the preference loss becomes a
loss on the policy itself (Eq. 7): a binary classification problem on preference pairs, no RL loop.

## 5. Mechanism ("DPO outline", Sec. 4, p5), transcribed
1. Sample completions y_1, y_2 ~ pi_ref(. | x) for every prompt x, label with human preferences to construct the
   offline dataset of preferences D = {x^(i), y_w^(i), y_l^(i)}_{i=1}^N.
2. Optimize the language model pi_theta to minimize L_DPO for the given pi_ref and D and desired beta.
pi_ref = pi_SFT when available; otherwise pi_ref is fit by maximum likelihood on the preferred completions.
Trained vs frozen: only pi_theta is trained; pi_ref is frozen (its log-probs are constants); no reward model network,
no value network. App. B code: `losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))`, with
`rewards = beta * (pi_logps - ref_logps).detach()` reported as the implicit reward.

## 6. Equations (printed numbers from the PDF, p3-4)
- Eq. 1, Bradley-Terry: `p^*(y_1\succ y_2 \mid x)=\frac{\exp\left(r^*(x, y_1)\right)}{\exp\left(r^*(x, y_1)\right) + \exp\left(r^*(x, y_2)\right)}`
- Eq. 2, reward-model loss: `\mathcal{L}_R(r_{\phi}, \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\bigl[\log \sigma(r_{\phi}(x, y_w)- r_{\phi}(x, y_l))\bigr]`
- Eq. 3, RL objective: `\max_{\pi_{\theta}}  \mathbb{E}_{x\sim \mathcal{D}, y\sim \pi_{\theta}(y \mid x)}\bigl[r_{\phi}(x, y)\bigr] - \beta\mathbb{D}_{\textrm{KL}}\bigl[\pi_{\theta}(y\mid x)\mid \mid \piref(y\mid x)\bigr]`
  ("beta is a parameter controlling the deviation from the base reference policy pi_ref, namely the initial SFT model").
- Eq. 4, optimal policy: `\pi_r(y\mid x) = \frac{1}{Z(x)}\piref(y\mid x)\exp\left(\frac{1}{\beta}r(x, y)\right)`, with
  `Z(x) =\sum_{y}\piref(y\mid x)\exp\left(\frac{1}{\beta}r(x, y)\right)` the partition function.
- Eq. 5, implicit reward (reparameterization): `r(x,y) =\beta \log \frac{\pi_r(y\mid x)}{\piref(y\mid x)} + \beta \log Z(x)`.
  Sec. 4 gradient paragraph (unnumbered): `\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\piref(y \mid x)}`
  "is the reward implicitly defined by the language model pi_theta and reference model pi_ref".
- Eq. 6: preference probability in terms of pi^* and pi_ref only (Z cancels).
- Eq. 7, DPO loss (p4):
  `\mathcal{L}_\text{DPO}(\pi_{\theta}; \piref) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_{\theta}(y_w\mid x)}{\piref(y_w\mid x)} - \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\piref(y_l\mid x)}\right)\right]`
  Source: "This way, we fit an implicit reward using an alternative parameterization, whose optimal policy is simply
  pi_theta." Gloss: y_w, y_l = preferred and dispreferred responses; each log-ratio is the implicit reward / beta;
  sigma = logistic; the loss is binary cross-entropy on "which response wins".
- Macro: `\piref` = `\pi_\text{ref}`. Gradient (unnumbered, Sec. 4): examples are weighted by
  sigma(r^_theta(x, y_l) - r^_theta(x, y_w)), "higher weight when reward estimate is wrong".

## 7. Config (App. B, p20)
| Setting | Value |
|---|---|
| beta | 0.1 default; 0.5 for TL;DR summarization |
| Optimizer, learning rate | RMSprop, 1e-6, linear warmup from 0 over 150 steps |
| Batch size | 64 |
| Models | GPT-2-large (IMDb), GPT-J SFT (TL;DR), Pythia-2.8B (Anthropic HH); "up to 6B parameters" (Sec. 1) |
Sec. 6.1 sweep: beta in {0.05, 0.1, 1, 5} vs PPO target KL in {3, 6, 9, 12}; 22 runs in total.

## 8. Results
- Abstract: DPO "exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves
  response quality in summarization and single-turn dialogue while being substantially simpler to implement and
  train."
- Sentiment (Sec. 6.1): "DPO's reward/KL tradeoff strictly dominates PPO", even PPO with ground-truth rewards.
- TL;DR (Sec. 6.2, p9): "DPO has a win rate of approximately 61% at a temperature of 0.0, exceeding the performance of
  PPO at ~57% at its optimal sampling temperature of 0.0" (GPT-4 judged, vs reference summaries). Human evaluation:
  DPO samples at temperature 0.25 "were preferred 58% times over PPO samples at temperature 0".

## 9. Limitations (stated) and later fixes
Sec. 7: out-of-distribution generalization vs an explicit reward function needs more study; reward over-optimization
in DPO is open; evaluated only up to 6B parameters; GPT-4 win rates depend on the prompt.

## 10. Outline feedback
- Eq. 7 for the DPO loss is correct. The implicit-reward relation is Eq. 5 (with the beta log Z(x) term); the clean
  form beta log(pi_theta / pi_ref) is unnumbered (Sec. 4, gradient paragraph; Sec. 5.1).
- v3 wording (printed): the abstract says DPO is "eliminating the need for sampling from the LM during fine-tuning or
  performing significant hyperparameter tuning". The phrase "fitting a reward model," survives only in the unprinted
  first argument of `\rev`. For the reward-model claim, quote v3: "This change-of-variables approach avoids fitting an
  explicit, standalone reward model" (Sec. 4, p4), or "DPO is able to bypass both fitting an explicit reward and
  performing RL to learn the policy using a single maximum likelihood objective" (Sec. 5.1, p5). Intro (p2, not
  revised): "without explicit reward modeling or reinforcement learning".
- "DPO skips RL by turning preferences into a classification loss" matches v3: "allowing us to solve the standard
  RLHF problem with only a simple classification loss" (abstract). Avoid "DPO has no reward"; v3 frames the policy as
  an implicit reward model.
- viz-llmmodels for DPO: trained = policy; frozen = reference; absent = reward model, value model; data = offline
  preference pairs (no sampling during training).
