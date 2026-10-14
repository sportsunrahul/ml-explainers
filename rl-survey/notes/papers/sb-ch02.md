# S&B Chapter 2: Multi-armed Bandits

## 1. Identity
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018; 2020 printing read). See
  sb-ch01.md for the full book identity. Book page = PDF page - 22.
- Chapter 2 "Multi-armed Bandits", book pp. 25-45 (PDF 47-67).
- Sections: 2.1 A k-armed Bandit Problem p.25; 2.2 Action-value Methods p.27; 2.3 The 10-armed Testbed p.28;
  2.4 Incremental Implementation p.30; 2.5 Tracking a Nonstationary Problem p.32; 2.6 Optimistic Initial Values
  p.34; 2.7 Upper-Confidence-Bound Action Selection p.35; 2.8 Gradient Bandit Algorithms p.37; 2.9 Associative
  Search (Contextual Bandits) p.41; 2.10 Summary p.42; Bibliographical and Historical Remarks p.44.

## 2. Core problem
Choose repeatedly among k actions; each choice returns a reward from a stationary distribution that depends on
the action. Maximize the expected total reward over a time period, "for example, over 1000 action selections"
(2.1, pp.25-26). The feedback is evaluative: it grades the action taken and does not say which action was best.

## 3. What prior methods did and why that hurt (per the source)
- Sophisticated balancing methods exist, "However, most of these methods make strong assumptions about
  stationarity and prior knowledge that are either violated or impossible to verify in most applications and in
  the full reinforcement learning problem" (2.1, pp.26-27).
- Greedy selection "spends no time at all sampling apparently inferior actions to see if they might really be
  better" (2.2, p.27).
- Gittins indices / Bayesian methods need "complete knowledge of the prior distribution of possible problems";
  neither the theory nor the tractability generalize to the full RL problem (2.10, p.43).

## 4. The one key idea
Simple rules (eps-greedy, optimistic starts, UCB, gradient bandit) that balance exploration and exploitation
"at all" beat pure exploitation; estimates are kept incrementally with the general form
NewEstimate <- OldEstimate + StepSize [Target - OldEstimate] (Eq. 2.4).

## 5. Mechanisms, step by step
1. Sample-average action values (Eq. 2.1). "If the denominator is zero, then we instead define Q_t(a) as some
   default value, such as 0." (p.27)
2. Greedy selection (Eq. 2.2), "with ties broken arbitrarily".
3. eps-greedy: "behave greedily most of the time, but every once in a while, say with small probability eps,
   instead select randomly from among all the actions with equal probability, independently of the action-value
   estimates" (pp.27-28). Asymptotically "the probability of selecting the optimal action converges to greater
   than 1 - eps" (p.28).
4. Incremental mean (Eq. 2.3), general form (Eq. 2.4), constant step size for nonstationary problems (Eq. 2.5).
5. Optimistic initial values: set Q_1(a) = +5 instead of 0; "Whichever actions are initially selected, the reward
   is less than the starting estimates; the learner switches to other actions, being 'disappointed'" (2.6, p.34).
6. UCB (Eq. 2.10): "If N_t(a) = 0, then a is considered to be a maximizing action." (p.36)
7. Gradient bandit: softmax over preferences (Eq. 2.11), preference update with a reward baseline (Eq. 2.12).

Box "A simple bandit algorithm" (2.4, p.32, PDF 54), transcribed:
```
Initialize, for a = 1 to k:
    Q(a) <- 0
    N(a) <- 0
Loop forever:
    A <- argmax_a Q(a)    with probability 1 - eps   (breaking ties randomly)
         a random action  with probability eps
    R <- bandit(A)
    N(A) <- N(A) + 1
    Q(A) <- Q(A) + (1/N(A)) [R - Q(A)]
```

## 6. Equations (transcribed from the PDF; all verified on the rendered pages)
- Action value, unnumbered (2.1, p.26): `q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]`.
  "the expected reward given that a is selected". Q_t(a) = estimate at step t.
- Eq. 2.1 (2.2, p.27), sample average:
  `Q_t(a) \doteq \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}`
  1_predicate = 1 if the predicate is true, else 0. "We call this the sample-average method".
- Eq. 2.2 (p.27), greedy: `A_t \doteq \operatorname*{argmax}_a Q_t(a)`.
- Unnumbered (2.4, p.31): `Q_n \doteq \frac{R_1 + R_2 + \cdots + R_{n-1}}{n-1}` (estimate after n-1 selections of
  one action; R_i = reward after its i-th selection).
- Eq. 2.3 (p.31, PDF 53), last line of the derivation from `Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n} R_i`:
  `Q_{n+1} = Q_n + \frac{1}{n}\Big[R_n - Q_n\Big]`
  "which holds even for n = 1, obtaining Q_2 = R_1 for arbitrary Q_1. This implementation requires memory only for
  Q_n and n".
- Eq. 2.4 (p.31): `\textit{NewEstimate} \leftarrow \textit{OldEstimate} + \textit{StepSize}\Big[\textit{Target} - \textit{OldEstimate}\Big]`
  "This update rule (2.3) is of a form that occurs frequently throughout this book. The general form is (2.4)".
  "The expression [Target - OldEstimate] is an error in the estimate. It is reduced by taking a step toward the
  'Target.' The target is presumed to indicate a desirable direction in which to move, though it may be noisy."
  Step size of (2.3) is 1/n; "In this book we denote the step-size parameter by alpha or, more generally, by
  alpha_t(a)." (p.32)
- Eq. 2.5 (2.5, p.32): `Q_{n+1} \doteq Q_n + \alpha\Big[R_n - Q_n\Big]`, "where the step-size parameter
  alpha in (0, 1] is constant".
- Eq. 2.6 (p.32): `Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i} R_i`.
  "sometimes called an exponential recency-weighted average" (p.33).
- Eq. 2.7 (p.33): `\sum_{n=1}^{\infty}\alpha_n(a) = \infty \quad\text{and}\quad \sum_{n=1}^{\infty}\alpha_n^2(a) < \infty`.
  Conditions for convergence with probability 1; met by 1/n, not by constant alpha.
- Eqs. 2.8-2.9 (Exercise 2.7, p.35): `\beta_n \doteq \alpha/\bar{o}_n`,
  `\bar{o}_n \doteq \bar{o}_{n-1} + \alpha(1-\bar{o}_{n-1})` for n > 0, with `\bar{o}_0 \doteq 0`
  (unbiased constant-step-size trick).
- Eq. 2.10 (2.7, p.35, PDF 57), UCB:
  `A_t \doteq \operatorname*{argmax}_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]`
  ln t = natural log; N_t(a) = times a was selected prior to t; "the number c > 0 controls the degree of
  exploration". "the square-root term is a measure of the uncertainty or variance in the estimate of a's value"
  (p.36).
- Eq. 2.11 (2.8, p.37, PDF 59), soft-max:
  `\Pr\{A_t = a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} \doteq \pi_t(a)`
  H_t(a) = preference; "if we add 1000 to all the action preferences there is no effect on the action
  probabilities". "Initially all action preferences are the same (e.g., H_1(a) = 0, for all a)".
- Eq. 2.12 (p.37):
  ```
  H_{t+1}(A_t) \doteq H_t(A_t) + \alpha\big(R_t - \bar{R}_t\big)\big(1 - \pi_t(A_t)\big),  and
  H_{t+1}(a)   \doteq H_t(a)   - \alpha\big(R_t - \bar{R}_t\big)\pi_t(a),             for all a \ne A_t
  ```
  "where alpha > 0 is a step-size parameter, and R-bar_t in R is the average of the rewards up to but not
  including time t (with R-bar_1 = R_1)". Footnote 1 (p.37): "In the empirical results in this chapter, the
  baseline R-bar_t also included R_t." "The R-bar_t term serves as a baseline with which the reward is compared.
  ... The non-selected actions move in the opposite direction."
- Eq. 2.13 (box "The Bandit Gradient Algorithm as Stochastic Gradient Ascent", p.38, PDF 60):
  `H_{t+1}(a) \doteq H_t(a) + \alpha\frac{\partial\,\mathbb{E}[R_t]}{\partial H_t(a)}`, with
  `\mathbb{E}[R_t] = \sum_x \pi_t(x) q_*(x)` (unnumbered).
- Unnumbered steps of the same box (pp.39-40, PDF 61-62):
  - `\frac{\partial\mathbb{E}[R_t]}{\partial H_t(a)} = \sum_x \big(q_*(x) - B_t\big)\frac{\partial\pi_t(x)}{\partial H_t(a)}`;
    "B_t, called the baseline, can be any scalar that does not depend on x ... because the gradient sums to zero
    over all the actions, `\sum_x \frac{\partial\pi_t(x)}{\partial H_t(a)} = 0`".
  - `= \mathbb{E}\Big[\big(R_t - \bar{R}_t\big)\frac{\partial\pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)\Big]`
    (choosing B_t = R-bar_t and substituting R_t for q_*(A_t), "permitted because E[R_t|A_t] = q_*(A_t)").
  - `\frac{\partial\pi_t(x)}{\partial H_t(a)} = \pi_t(x)\big(\mathbb{1}_{a=x} - \pi_t(a)\big)` (proved on p.40 by
    the quotient rule).
  - `= \mathbb{E}\big[\big(R_t - \bar{R}_t\big)\big(\mathbb{1}_{a=A_t} - \pi_t(a)\big)\big]`.
  - Sampled: `H_{t+1}(a) = H_t(a) + \alpha\big(R_t - \bar{R}_t\big)\big(\mathbb{1}_{a=A_t} - \pi_t(a)\big)`, for all
    a, "which you may recognize as being equivalent to our original algorithm (2.12)" (p.39).

## 7. Config / example settings
10-armed testbed (2.3, pp.28-29, PDF 50-51):

| Setting | Value | Loc |
|---|---|---|
| Arms k | 10 | p.28 |
| Problems | "a set of 2000 randomly generated k-armed bandit problems" | p.28 |
| q*(a), a = 1..10 | normal, mean 0, variance 1 | p.29; Fig 2.1 caption p.28 |
| R_t given A_t | normal, mean q*(A_t), variance 1 | p.29 |
| Steps per run | 1000 | p.29 |
| Runs averaged | 2000 independent runs, each with a different bandit problem | p.29 |

Per-figure settings:

| Figure | Methods and settings | Where the numbers are printed |
|---|---|---|
| 2.2 (p.29) | greedy, eps = 0.01, eps = 0.1; sample averages, initial estimate 0; 2000 runs | body text p.29 and caption |
| 2.3 (p.34) | greedy with Q_1(a) = +5 (eps = 0) vs eps-greedy with Q_1(a) = 0, eps = 0.1; both constant alpha = 0.1; % optimal action | Q_1 = +5 in body text; eps = 0.1 only in curve label; alpha = 0.1 in caption; 2000 tasks in Exercise 2.6 (p.35) |
| 2.4 (p.36) | UCB c = 2 vs eps-greedy eps = 0.1; average reward | c = 2 and eps = 0.1 only as curve labels; estimate method not stated |
| 2.5 (p.38) | gradient bandit, alpha = 0.1 and 0.4, with and without baseline; q*(a) ~ normal mean +4, unit variance; % optimal action | mean +4 in body text p.37; alphas only as curve labels; baseline includes R_t (footnote 1, p.37) |
| 2.6 (p.42) | parameter study, average reward over first 1000 steps, eps-greedy / UCB / gradient bandit / greedy with optimistic init (alpha = 0.1); parameters varied by factors of two on a log scale | text p.42; tick labels not extracted |

## 8. Results and findings worth remembering
- Evaluative vs instructive (ch. 2 opening, p.25, PDF 47): "The most important feature distinguishing
  reinforcement learning from other types of learning is that it uses training information that evaluates the
  actions taken rather than instructs by giving correct actions. ... Purely evaluative feedback indicates how good
  the action taken was, but not whether it was the best or the worst action possible. Purely instructive
  feedback, on the other hand, indicates the correct action to take, independently of the action actually taken.
  ... evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the
  action taken."
- Exploration pays later (2.1, p.26): "Exploitation is the right thing to do to maximize the expected reward on
  the one step, but exploration may produce the greater total reward in the long run." "Reward is lower in the
  short run, during exploration, but higher in the long run".
- Figure 2.2 (pp.29-30, PDF 51-52): greedy "improved slightly faster than the other methods at the very
  beginning, but then leveled off at a lower level. It achieved a reward-per-step of only about 1, compared with
  the best possible of about 1.54 on this testbed." Greedy "found the optimal action in only approximately
  one-third of the tasks. In the other two-thirds, its initial samples of the optimal action were disappointing,
  and it never returned to it." "The eps = 0.1 method explored more, and usually found the optimal action
  earlier, but it never selected that action more than 91% of the time. The eps = 0.01 method improved more
  slowly, but eventually would perform better than the eps = 0.1 method on both performance measures shown in the
  figure."
- Variance and nonstationarity (p.30): with reward variance 10, eps-greedy "should fare even better"; with zero
  variance greedy "might actually perform best"; "nonstationarity is the case most commonly encountered in
  reinforcement learning. Even if the underlying task is stationary and deterministic, the learner faces a set of
  banditlike decision tasks each of which changes over time as learning proceeds and the agent's decision-making
  policy changes."
- Figure 2.3 (p.34): "Initially, the optimistic method performs worse because it explores more, but eventually it
  performs better because its exploration decreases with time." It is "a simple trick that can be quite effective
  on stationary problems, but it is far from being a generally useful approach"; "its drive for exploration is
  inherently temporary" (pp.34-35). Exercise 2.6: early "oscillations and spikes" in the optimistic curve.
- Figure 2.4 caption (p.36): "UCB generally performs better than eps-greedy action selection, except in the first
  k steps, when it selects randomly among the as-yet-untried actions." Exercise 2.8 (p.36): "a distinct spike in
  performance on the 11th step ... Hint: If c = 1, then the spike is less prominent."
- eps-greedy explores "indiscriminately, with no preference for those that are nearly greedy or particularly
  uncertain" (p.35). UCB: "all actions will eventually be selected, but actions with lower value estimates, or
  that have already been selected frequently, will be selected with decreasing frequency over time." (p.36)
- Figure 2.5 (p.37): "This shifting up of all the rewards has absolutely no effect on the gradient bandit
  algorithm because of the reward baseline term, which instantaneously adapts to the new level. But if the
  baseline were omitted (that is, if R-bar_t was taken to be constant zero in (2.12)), then performance would be
  significantly degraded, as shown in the figure."
- SGA box conclusion (p.40): "the expected update of the gradient bandit algorithm is equal to the gradient of
  expected reward, and thus that the algorithm is an instance of stochastic gradient ascent." "The choice of the
  baseline does not affect the expected update of the algorithm, but it does affect the variance of the update
  and thus the rate of convergence (as shown, for example, in Figure 2.5)."
- Figure 2.6 / summary (pp.42-43): inverted-U parameter curves; algorithms "fairly insensitive, performing well
  over a range of parameter values varying by about an order of magnitude. Overall, on this problem, UCB seems to
  perform best." Thompson sampling "often performs similarly to the best of the distribution-free methods"
  (p.43).
- History (bib remarks, pp.44-45): sequential design of experiments, Thompson (1933, 1934) and Robbins (1952);
  "The term action value is due to Watkins (1989). The first to use eps-greedy methods may also have been Watkins
  (1989, p. 187)"; UCB as presented "is called UCB1 in the literature and was first developed by Auer,
  Cesa-Bianchi and Fischer (2002)"; gradient bandits are "a special case of the gradient-based reinforcement
  learning algorithms introduced by Williams (1992) that later developed into the actor-critic and policy-gradient
  algorithms"; "The term soft-max for the action selection rule (2.11) is due to Bridle (1990). This rule appears
  to have been first proposed by Luce (1959)"; optimistic initialization in RL by Sutton (1996); associative
  search term by Barto, Sutton, and Brouwer (1981).

## 9. Limitations the source states
- UCB "is more difficult than eps-greedy to extend beyond bandits": nonstationary problems and large state spaces
  with function approximation; "the idea of UCB action selection is usually not practical" there (p.36).
- Optimistic starts are not suited to nonstationary problems (pp.34-35); sample averages "may perform poorly on
  nonstationary problems" (Exercise 2.7, p.35).
- All methods here are "far from a fully satisfactory solution to the problem of balancing exploration and
  exploitation" (p.43).
- Later work named by the source: the gradient bandit leads to actor-critic and policy-gradient methods (Williams
  1992; book ch. 13) (bib 2.8, p.44).

## 10. Outline feedback (ch-bandits)
1. The general form "NewEstimate <- OldEstimate + StepSize [Target - OldEstimate]" is Eq. 2.4, not Eq. 2.3. Eq.
   2.3 is the incremental mean (step 1/n). Print "Eq. 2.4" next to the general form, and add Eqs. 2.4 and 2.5 to
   the chapter's Sources line.
2. Index shift inside S&B: ch. 2 writes the reward for A_t as R_t ("We denote the action selected on time step t
   as A_t, and the corresponding reward as R_t", p.26; Eq. 2.12 uses R_t). R_{t+1} starts in ch. 3 (footnote 4,
   p.48). Say this where the page first uses R_{t+1}, or keep R_t in the bandit chapter as the book does.
3. viz-testbed: the book averages 2000 runs of 1000 steps. Text findings to check the computed curves against:
   greedy about 1 vs best possible about 1.54; greedy finds the optimal arm in about one-third of tasks; eps 0.1
   never above 91% optimal; eps 0.01 "eventually" better on both measures (an asymptotic claim that may not show by
   step 1000). Our arithmetic: 91% = 1 - 0.1 + 0.1/10, because the random pick includes the greedy arm.
4. UCB c = 2 appears only as a curve label in Figure 2.4. The book compares UCB with eps 0.1 on average reward only
   and does not state the value-estimate method. To reproduce the step-11 spike (Exercise 2.8), treat N_t(a) = 0
   as maximizing (p.36), so the first k steps try each arm once.
5. Optimistic starts (Figure 2.3) use constant alpha = 0.1 for both curves, Q_1 = +5 greedy vs Q_1 = 0 with
   eps = 0.1. Use these if the paragraph quotes numbers.
6. Gradient bandit PyTorch exercise: the text baseline excludes R_t (R-bar_1 = R_1), but the book's experiments
   included R_t (footnote 1, p.37). State which one the exercise uses. The autograd identity holds either way,
   because the baseline is a constant with respect to H. Figure 2.5 settings: mean +4, unit variance, alpha 0.1
   and 0.4, H_1(a) = 0.
7. The SGA derivation has one numbered equation (2.13, p.38). Its sampled-update line and the softmax-derivative
   identity are unnumbered (pp.39-40); cite them as "S&B 2.8 box, pp.38-40". The book's own baseline argument
   ("the gradient sums to zero over all the actions", p.39) is the cleanest plant for the ch. 10 baseline proof.
8. Confusion line "eps-greedy explores even hopeless arms (UCB does not)": per p.36, UCB still selects every arm
   eventually, but low-value and often-tried arms "with decreasing frequency over time". Reword to "UCB explores
   them less and less".
9. Applications named in 2.1 are slot machines and a doctor choosing experimental treatments (p.26). The A/B-test
   and recommendation framing is the page's own; "contextual bandits" is the book's term for associative search
   (2.9, p.41).
