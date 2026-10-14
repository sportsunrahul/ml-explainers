# InstructGPT (Ouyang et al. 2022)

## 1. Identity
- Title: Training language models to follow instructions with human feedback. arXiv 2203.02155, version read v1
  (2022-03-04; only version). PDF 68 pages.
- Authors: Long Ouyang (first), Ryan Lowe (last; 20 authors).
- Sections used: 1 Introduction, 3 Methods and experimental details (3.1 High-level methodology, 3.5 Models),
  4 Results (4.1 API distribution), 5.3 Limitations, App. C Additional model details (C.2 RM, C.4 RLHF training),
  App. E.6-E.7 (KL and pretraining-mix sweeps).

## 2. Core problem
Bigger language models do not follow user intent better; they can be untruthful, toxic or unhelpful. Align GPT-3
with what users want using human demonstrations and human rankings.

## 3. What prior methods did and why that hurt (per the source)
GPT-3 is trained to predict internet text, so its behavior is "poorly characterized" (Sec. 3.5). Few-shot prompting
helps but less than fine-tuning (Sec. 4.1). Models fine-tuned on public NLP task collections (FLAN, T0) do worse than
SFT on the API prompt distribution: "Public NLP datasets are not reflective of how our language models are used."

## 4. Key idea
The RLHF recipe of Ziegler et al. and Stiennon et al. at GPT-3 scale on a broad prompt distribution: SFT, a reward
model from rankings, then PPO against the reward model with a per-token KL penalty to the SFT model, plus optional
pretraining-gradient mixing (PPO-ptx).

## 5. Mechanism
The three steps (Sec. 3.1, p6), transcribed:
1. Collect demonstration data, and train a supervised policy. Labelers demonstrate the desired behavior on the
   prompt distribution; fine-tune a pretrained GPT-3 model on it with supervised learning.
2. Collect comparison data, and train a reward model. Labelers indicate which output they prefer for a given input;
   train a reward model to predict the human-preferred output.
3. Optimize a policy against the reward model using PPO. Use the RM output as a scalar reward; fine-tune the
   supervised policy with PPO.
"Steps 2 and 3 can be iterated continuously" (p6).

RL environment (Sec. 3.5, p9): "The environment is a bandit environment which presents a random customer prompt
and expects a response to the prompt. Given the prompt and response, it produces a reward determined by the reward
model and ends the episode." A per-token KL penalty from the SFT model is added at each token. "The value function
is initialized from the RM."

What is trained and what is frozen during RL: policy pi^RL_phi trained (initialized from SFT with pretraining mix);
value function trained (6B, initialized from the 6B RM); reward model r_theta frozen; KL reference = the SFT init
model, frozen (App. C.4: "These models are also used to compute the KL reward").

Reward model (Sec. 3.5, p8; App. C.2, p41): starts from the SFT model with the unembedding layer removed and outputs
a scalar. Labelers rank K = 4 to K = 9 responses per prompt, giving (K choose 2) comparisons; all comparisons from one
prompt form a single batch element (avoids overfitting). One 6B RM is used for all PPO models: 175B RM training
"could be unstable". After training, the RM is shifted so labeler demonstrations score 0 on average.

## 6. Equations
- Eq. 1 (Sec. 3.5, p8), reward model loss:
  `\operatorname{loss}\left(\theta \right)=-\frac{1}{{K \choose 2}}E_{\left(x, y_{w}, y_{l}\right) \sim D}\left[\log \left(\sigma\left(r_{\theta}\left(x, y_{w}\right)-r_{\theta}\left(x, y_{l}\right)\right)\right)\right]`
  Source: "r_theta(x, y) is the scalar output of the reward model for prompt x and completion y with parameters theta,
  y_w is the preferred completion out of the pair of y_w and y_l, and D is the dataset of human comparisons."
  Gloss: sigma = logistic function; the difference in rewards is the log-odds that y_w is preferred (Bradley-Terry).
- Eq. 2 (Sec. 3.5, p9), RL objective (PPO-ptx):
  `\operatorname{objective}\left(\phi\right)= & E_{\left(x, y\right) \sim D_{\pi_{\phi}^{\mathrm{RL}}}}\left[r_{\theta}(x, y)-\beta \log \left(\pi_{\phi}^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right] + \\ & \gamma E_{x \sim D_\textrm{pretrain}}\left[\log(\pi_{\phi}^{\mathrm{RL}}(x))\right]`
  Source: "The KL reward coefficient, beta, and the pretraining loss coefficient, gamma, control the strength of the
  KL penalty and pretraining gradients respectively. For "PPO" models, gamma is set to 0." Gloss: x = prompt,
  y = sampled response; r_theta = frozen RM score; the log-ratio term is a sample of the KL from the SFT model
  (written per sequence; the text says the penalty is applied per token); D_pretrain = GPT-3 pretraining data.
  "Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models."
- Notation trap: gamma in Eq. 2 is the pretraining-loss coefficient (27.8), not a discount. No discount is used:
  "No discount is applied when estimating the generalized advantage" (App. C.4, p42).

## 7. Config
| Setting | Value | Location |
|---|---|---|
| KL coefficient beta | 0.02 | App. C.4, p42 |
| Pretraining coefficient gamma (PPO-ptx) | 27.8; 0 for "PPO" models | App. C.4 p42; Sec. 3.5 p9 |
| Pretraining examples | 8 times the number of RL episodes | App. C.4, p42 |
| RL episodes | 256k (about 31k unique prompts) | App. C.4, p42 |
| PPO batch / minibatch | 512 / 64, one inner epoch | App. C.4, p42 |
| PPO clip ratio; rollout temperature | 0.2; 1 | App. C.4, p42 |
| Reward model / value function size | 6B / 6B (value initialized from RM) | Sec. 3.5; App. C.4 |
| RM training | 1 epoch, lr 9e-6, batch 64 prompts (up to 2,304 comparisons) | App. C.2, p41 |
| SFT | 16 epochs, cosine decay, residual dropout 0.2 | Sec. 3.5, p8 |
| Policy sizes | 1.3B, 6B, 175B (GPT-3 architecture) | Sec. 3.5; App. C |
| Labelers | about 40 contractors | Sec. 5.3, p19 |
| KL sweep | optimal coefficient "around 0.01 and 0.02"; even 2.0 ("100 times of the default value") does not fix the NLP regressions | App. E.7 p56; App. E.6 p53 |

## 8. Results
- Abstract (p1) verbatim: "In human evaluations on our prompt distribution, outputs from the 1.3B parameter
  InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters." (Sec. 1, p3,
  says "despite having over 100x fewer parameters".)
- Sec. 4.1 (p11): "175B InstructGPT outputs are preferred to GPT-3 outputs 85 +- 3% of the time, and preferred
  71 +- 4% of the time to few-shot GPT-3."
- Truthfulness: on TruthfulQA, truthful and informative answers "about twice as often as GPT-3"; closed-domain
  hallucination 21% vs 41% (Sec. 1, p3).
- Alignment tax: PPO regresses on SQuAD, DROP, HellaSwag, WMT 2015 Fr-En; PPO-ptx "can greatly reduce the
  performance regressions" (Sec. 1, p3-4; Sec. 4.2).

## 9. Limitations (stated) and later fixes
- Aligned to "about 40 contractors", primarily English-speaking (Sec. 5.3, p19). "Our models are neither fully aligned
  nor fully safe" (Sec. 5.3).
- RM over-optimization is the stated reason for the KL penalty (Sec. 3.5). DPO (2305.18290) solves the same
  KL-constrained objective without RL; GRPO (2402.03300) removes the value model.

## 10. Outline feedback
- Mapping table "gamma = 1": cite App. C.4 ("No discount is applied when estimating the generalized advantage"), and
  warn that InstructGPT's own gamma (Eq. 2) is the pretraining-mix weight 27.8. Rename it on the page (for example
  gamma_ptx) or gloss it, so readers do not take it for the discount.
- "the reward usually arrives once, at the end": the paper says it directly ("a bandit environment ... ends the
  episode"); quote it.
- Eq. 2 shows a sequence log-ratio, while the text applies the KL penalty per token. Say both, so the toy (per-token
  KL in the PyTorch exercise) matches the source.
- viz-llmmodels for PPO-RLHF: trained = policy, value (6B, init from RM); frozen = reward model (6B), SFT reference.
