# DeepSeekMath and GRPO (Shao et al. 2024)

## 1. Identity
- Title: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv 2402.03300,
  version read v3 (2024-04-27); v1 2024-02-05. PDF 30 pages.
- Authors: Zhihong Shao (first), Daya Guo (last; 11 authors).
- Sections used: 4 Reinforcement Learning (4.1 Group Relative Policy Optimization: 4.1.1 From PPO to GRPO,
  4.1.2 Outcome Supervision RL with GRPO, 4.1.3 Process Supervision RL with GRPO, 4.1.4 Iterative RL with GRPO;
  4.2 Training and Evaluating DeepSeekMath-RL), 5.2 Insights of RL, App. A.1 (A.1.6 GRPO gradient).

## 2. Core problem
PPO for LLMs needs a value model about as large as the policy, and with a reward only at the last token it is hard
to train a value function that is accurate at each token.

## 3. What prior methods did and why that hurt (per the source, Sec. 4.1.1, p13)
"As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a
substantial memory and computational burden. ... While in the LLM context, usually only the last token is assigned a
reward score by the reward model, which may complicate the training of a value function that is accurate at each
token."

## 4. Key idea
Drop the critic. For each question, sample a group of G outputs; use the group's mean reward as the baseline and
the group's std as the scale; move the KL penalty from the reward into the loss.

## 5. Mechanism
Algorithm 1, Iterative GRPO (p14), transcribed:
- Input: initial policy model pi_theta_init; reward models r_phi; task prompts D; hyperparameters epsilon, beta, mu.
1. policy model pi_theta <- pi_theta_init
2. for iteration = 1, ..., I do
3.   reference model pi_ref <- pi_theta
4.   for step = 1, ..., M do
5.     Sample a batch D_b from D
6.     Update the old policy model pi_theta_old <- pi_theta
7.     Sample G outputs {o_i}_{i=1}^G ~ pi_theta_old(. | q) for each question q in D_b
8.     Compute rewards {r_i}_{i=1}^G for each sampled output o_i by running r_phi
9.     Compute A^_{i,t} for the t-th token of o_i through group relative advantage estimation.
10.    for GRPO iteration = 1, ..., mu do
11.      Update the policy model pi_theta by maximizing the GRPO objective (Equation 21)
12.  Update r_phi through continuous training using a replay mechanism.
- Output: pi_theta
Note: the printed line 11 cites "Equation 21", which is the gradient coefficient GC_GRPO in App. A.1.6, not the
objective (Eq. 3).

Outcome supervision (Sec. 4.1.2, p14): "these rewards are normalized by subtracting the group average and dividing by
the group standard deviation. Outcome supervision provides the normalized reward at the end of each output o_i and
sets the advantages A^_{i,t} of all tokens in the output as the normalized reward".
Process supervision (Sec. 4.1.3): a process reward model scores each step; A^_{i,t} = sum of normalized step rewards
from the following steps.
Iterative RL (Sec. 4.1.4): retrain the RM on new policy samples with a replay of 10% historical data; reset the
reference model to the current policy.

Constants vs gradient: pi_theta_old (sampling policy), pi_ref, and r_phi are fixed during the update; A^_{i,t} is a
constant computed from the group rewards; gradients flow through pi_theta in the ratio and in the KL term.
Sec. 4.2: "The policy model only has a single update following each exploration stage", so mu = 1 and
pi_theta_old = pi_theta at the update; the ratio is 1 and the clip never binds there. App. A.1.6 analyses GRPO under
exactly that assumption ("assume pi_theta_old = pi_theta for simplified analysis").

## 6. Equations (printed numbers from the PDF)
- Eq. 1 (Sec. 4.1.1, p11), PPO surrogate:
  `\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}, \text{clip} \left( \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right)  A_{t} \right]`
  A_t from GAE with a learned value function V_psi.
- Eq. 2 (p13), KL penalty inside the PPO reward:
  `r_{t} = r_\phi(q, o_{\le t}) - \beta \log\frac{\pi_{\theta}(o_{t}|q, o_{<t})}{\pi_{ref}(o_{t}|q, o_{<t})}`
  ("where r_phi is the reward model, pi_ref is the reference model, which is usually the initial SFT model, and beta is
  the coefficient of the KL penalty"). Notation: r_t is the reward for token o_t (deep-RL indexing).
- Eq. 3 (p13), GRPO objective:
  `\mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \\ & \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right)  \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]\right\}`
  Source: "where epsilon and beta are hyper-parameters, and A^_{i,t} is the advantage calculated based on relative
  rewards of the outputs inside each group only". Gloss: q = question; o_i = i-th sampled output, o_{i,t} its t-th
  token; G = group size; the ratio is per token; per-output averaging by 1/|o_i|, then 1/G over the group.
- Eq. 4 (p14), KL estimator:
  `\mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1`
  Source: "we estimate the KL divergence with the following unbiased estimator (Schulman, 2020) ... which is guaranteed
  to be positive." Gloss: per token, with ratio rho = pi_ref / pi_theta, the estimate is rho - log rho - 1 >= 0.
- Outcome advantage (UNNUMBERED, inline in Sec. 4.1.2, p14):
  `\hat{A}_{i, t} = \widetilde{r}_i = \frac{r_i- {\rm mean}(\mathbf{r})}{{\rm std}(\mathbf{r})}`, with
  r = {r_1, ..., r_G}. Every token of output i gets the same advantage.
- App. A.1.6: Eq. 19 (objective with pi_old = pi_theta), Eq. 20 (gradient), Eq. 21 GC_GRPO = A^_{i,t} +
  beta (pi_ref / pi_theta - 1).

## 7. Config (Sec. 4.2, p15)
| Setting | Value |
|---|---|
| Start model | DeepSeekMath-Instruct 7B |
| RL data | about 144K chain-of-thought questions from GSM8K and MATH SFT data |
| Reward model | trained from DeepSeekMath-Base 7B, learning rate 2e-5 |
| Policy learning rate | 1e-6 |
| KL coefficient beta | 0.04 |
| Group size G | 64 outputs per question |
| Max length; training batch size | 1024; 1024 |
| Policy updates per exploration stage (mu) | 1 ("a single update") |
| Clip epsilon | not stated in the paper |
| Std estimator, all-equal-rewards case (std = 0) | not stated |

## 8. Results
- Sec. 4.2 (p15) and Table 5 (p12): "DeepSeekMath-RL 7B attains accuracies of 88.2% and 51.7% on GSM8K and MATH,
  respectively, utilizing chain-of-thought reasoning", and it "outperforms DeepSeekMath-Instruct 7B across all
  evaluation metrics" despite RL data from GSM8K and MATH only.
- Quote on the value function (Sec. 4.1.1, p13): GRPO "obviates the need for additional value function approximation
  as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same
  question, as the baseline." Figure 4 caption (p13): "GRPO foregoes the value model, instead estimating the baseline
  from group scores, significantly reducing training resources."
- Figure 4 (p13, diagram): PPO = policy and value models trained; reference and reward models frozen. GRPO = policy
  trained; reference and reward models frozen; no value model.

## 9. Limitations (stated) and later fixes
- Outcome supervision gives one reward per output, "which may not be sufficient and efficient to supervise the policy
  in complex mathematical tasks" (Sec. 4.1.3), motivating process supervision.
- "the old reward model may not be sufficient to supervise the current policy model" (Sec. 4.1.4) -> iterative RL.
- DeepSeek-R1 (2501.12948) reuses GRPO with rule-based rewards instead of a neural reward model for reasoning tasks.

## 10. Outline feedback
- "GRPO advantage and objective (DeepSeekMath, as numbered)": the objective is Eq. 3 and the KL estimator Eq. 4, but
  the outcome advantage has no number (inline, Sec. 4.1.2, p14). Cite it by section, or use R1 v2 Eq. 3, which numbers
  the same normalized advantage. Do not cite Algorithm 1's "Equation 21" as the objective.
- Toy fidelity: DeepSeekMath uses a per-token ratio with 1/|o_i| averaging; R1 v2 writes a per-output ratio
  pi(o_i | q) without 1/|o_i| (see r1.md). Pick one form for viz-grpo and the exercises and say which.
- With mu = 1 (Sec. 4.2) the clip is inert and the update is a REINFORCE-style step with a group baseline plus the KL
  gradient, matching the outline's "REINFORCE with a baseline" framing; App. A.1.6 Eq. 20 shows this gradient.
- viz-grpo uses 8 answers; the papers use G = 64 (DeepSeekMath) and 16 (R1-Zero). State the real values in the
  caption. Also decide and state what the toy does when all 8 rewards are equal (std = 0); neither source says.
- KL coefficients differ by paper: 0.02 (InstructGPT), 0.04 (DeepSeekMath), 0.001 (R1-Zero).
