# van Hasselt et al. 2018: Deep Reinforcement Learning and the Deadly Triad

## 1. Identity
- arXiv 1812.02648v1 (the only version). Submitted Thu, 6 Dec 2018 16:36:20 UTC (abs.html submission history;
  citation_date 2018/12/06). PDF page 1 stamp: "arXiv:1812.02648v1 [cs.AI] 6 Dec 2018".
- Authors: Hado van Hasselt (first; DeepMind), Yotam Doron, Florian Strub (University of Lille, DeepMind), Matteo
  Hessel, Nicolas Sonnerat, Joseph Modayil (last; DeepMind).
- Venue: none printed. The LaTeX uses the nips_2018 style with [final], whose notice string is commented out, so no
  conference line appears (src/nips_2018.sty lines 60-75). Treat it as an arXiv preprint.
- Layout: 13 pages. Intro (p.1-2, unnumbered); 1 The deadly triad in deep reinforcement learning (p.2); 2 Building
  intuition (p.2-3); 3 Hypotheses (p.3-5); 4 Evaluation (p.5; 4.1 p.5; 4.2 p.6; 4.3 p.7; 4.4 p.7-8); 5 Discussion (p.8);
  Appendix A Experiment details (p.11); B Training curve at early stages (p.11); C Visualising overestimation
  distribution (p.12); D Scores breakdown on individual games (p.13, Table 1).
- Notation: S&B convention. Eq. 1 uses R_{t+1} after S_t; the n-step return is
  G^(n)_t = R_{t+1} + ... + gamma^{n-1} R_{t+n} + gamma^n v(S_{t+n}) (p.4). q' = target network.

## 2. Core problem
Theory says function approximation + bootstrapping + off-policy learning can diverge, yet DQN combines all three
and works. "In this paper, we conduct an empirical examination into when the triad becomes deadly." (p.2)

## 3. What prior methods did and why that hurt (per the source)
- p.1-2: "Partial solutions have been proposed (Baird, 1995; Precup and Sutton, 2001; Sutton et al., 2008; Maei et al.,
  2009; Maei, 2011; Sutton et al., 2016), but they have mostly not been extended to non-linear function approximation,
  and have not been widely adopted in practice."
- p.2: DQN uses deep networks (function approximation), Q-learning (off-policy), and experience replay, which samples
  "according to a mixture of past policies rather than the current policy. This causes the updates to be even more
  off-policy." It bootstraps with one-step Q-learning. "Despite combining all these components of the deadly triad,
  DQN successfully learnt to play many Atari 2600 games".
- p.3: "the examples of divergence are typically contrived"; with deep networks "there is little guidance on whether
  divergence is common, and if the deadly triad is to blame."

## 4. The one key idea
Each triad component is a dial, not a switch (Sec. 1, p.2): bootstrapping via the multi-step length n; function
approximation via network size; off-policy via prioritized replay. Turn each dial across many DQN variants on Atari
and measure how often the value estimates leave the possible range.

## 5. Mechanism (experimental design), step by step
1. Toy intuition (Sec. 2): the Tsitsiklis and Van Roy w, 2w example (Figure 1a).
2. Six hypotheses (Sec. 3, quoted in Sec. 6 below).
3. Variants of DQN on the Arcade Learning Environment, same preprocessing as Mnih et al. (2015) (Sec. 4, Appendix A).
4. Vary: 4 bootstrap targets x n in {1, 3, 10} x prioritization alpha in {0, 1/2, 1, 2}, with (beta = 0.4) or without
   (beta = 0) importance-sampling correction x 4 network sizes: "a total of 336 parameter settings per game, for 57
   different games" (p.5). "For reliable results, we ran 3 distinct replications of each experiment." (p.5)
5. Run each configuration "for roughly 20M frames on a single CPU core (which requires a few days per configuration)"
   (p.5).
6. Track the maximal absolute action value estimate ("maximal |Q|") over roughly 50K-100K frame intervals (p.5).
   A run soft-diverges if it exceeds 100 (definition below).
7. Report soft-divergence rates per bootstrap type and per dial (Figures 2, 3), and control performance as median
   human-normalised score across 57 games (Figure 4).

## 6. Equations and definitions (LaTeX source; printed numbers from the PDF)
- Eq. 1, Sec. 2, p.2 (the only numbered equation): `\Delta\mathbf{w} ~~\propto~~ (R_{t+1} + \gamma v_{\mathbf{w}}(S_{t+1}) - v_{\mathbf{w}}(S_t))\nabla_{\mathbf{w}}v_{\mathbf{w}}(S_t)`
  "where v_w(S_{t+1}) is used to estimate the unknown remainder of the return after state S_{t+1}."
- Toy example (Sec. 2, p.3): "each state is described by a single scalar feature phi, such that phi(s_1) = 1 and
  phi(s_2) = 2"; v(s) = w x phi; "All rewards are 0, and therefore the value predictions would be perfect with
  w_* = 0." Off-policy (update s_1 only): `\Delta w \propto \gamma 2w - w = (2\gamma - 1)w`; "For a discount
  gamma > 1/2, we have 2 gamma > 1 and so any weight w != 0 will be updated away from zero, regardless of its current
  value: w diverges. Even if we update s_2 as often as s_1, this would be less often than under on-policy updating,
  and we will still see divergence for large enough gamma (Figure 1b)." On-policy: "the value of state s_2 is updated,
  in expectation, multiple times for each update to the value of state s_1. Then, the weight w will converge to its
  optimal setting, w_* = 0." Figure 1a draws the Tsitsiklis-Van Roy MDP: w -> 2w; 2w loops with 1 - epsilon and
  terminates with epsilon. gamma, epsilon and the step size used for Figures 1b-1c are not stated.
- Extra parameter (p.3): `v(s_i) = w(\phi(s_i) + u)` with u learnable: "values first seemingly diverge, then recover
  and converge to the optimum v(s_1) = v(s_2) = 0 (see Figure 1c)." Figure 1 caption: "Both systems are initialized
  to the same point (with u = 0) and optimized identically. The number of learning updates are shown on the x-axis,
  and the values are shown on the y-axis in log-scale."
- Update family (Sec. 3, p.3-4): `\Delta\theta \propto (G^{(n)}_t - q(S_t, A_t))\nabla_{\theta} q(S_t, A_t)` with
  `G^{(n)}_t = R_{t+1} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n v(S_{t+n})`.
- Four bootstrap targets (p.4), as printed with tags:
  `v(s) = q(s, \arg\max_a q(s, a)) = \max_a q(s, a)` (Q-learning)
  `v(s) = q'(s, \arg\max_a q'(s, a)) = \max_a q'(s, a)` (target Q-learning)
  `v(s) = q(s, \arg\max_a q'(s, a))` (inverse double Q-learning)
  `v(s) = q'(s, \arg\max_a q(s, a))` (double Q-learning)
  Inverse double Q is new here: it reduces overestimation but bootstraps on the online network (p.4).
- Prioritization (p.5): `p_k \propto \lvert G^{(n)}_k - q(S_k, A_k) \rvert^\alpha`; alpha = 0 is uniform replay;
  importance-sampling correction `1/(N p_k)^\beta`, N = replay size. "We can modulate how off-policy the updates are
  with alpha and beta."
- Soft divergence (definition, Sec. 4, p.5): "Because the rewards are clipped to [-1, 1], and because the discount
  factor gamma is 0.99, the maximum absolute true value in each game is bounded by 1 + gamma + gamma^2 + ... =
  1/(1-gamma) = 100 (and realistically attainable values are typically much smaller). Therefore, values for which
  |q| > 100 are unrealistic. We call this phenomenon soft divergence."
- Unbounded divergence (as used, Sec. 4.1, p.5): values growing without bound; operationally, NaNs: "they never
  became unbounded (more precisely, they never produced floating point NaNs)." Abstract: learning "can diverge with
  the value estimates becoming unbounded".
- Hypotheses, verbatim with printed numbers:
  - Hypothesis 1 (Deep divergence), p.3: "Unbounded divergence is uncommon when combining Q-learning and conventional
    deep reinforcement learning function spaces."
  - Hypothesis 2 (Target networks), p.4: "There is less divergence when bootstrapping on separate networks."
  - Hypothesis 3 (Overestimation), p.4: "There is less divergence when correcting for overestimation bias."
  - Hypothesis 4 (Multi-step), p.4: "Longer multi-step returns will diverge less easily."
  - Hypothesis 5 (Capacity), p.5: "Larger, more flexible networks will diverge less easily."
  - Hypothesis 6 (prioritisation), p.5 (printed lowercase): "Stronger prioritisation of updates will diverge more
    easily."

## 7. Measurement setup and config (Sec. 4 p.5; Appendix A p.11)
| item | value |
|---|---|
| domain | Atari 2600 via the Arcade Learning Environment; "the full selection of 57 Atari games" |
| varied | bootstrap target (4); n = 1, 3, 10; alpha in {0, 1/2, 1, 2} with beta = 0.4 or beta = 0; network small/medium/large/extra-large |
| settings per game | 336 |
| replications | 3 ("Each configuration was run three times, with different random seeds.") |
| run length | roughly 20M frames, single CPU core |
| logging interval | roughly 50K-100K frames |
| preprocessing | as DQN: "downsampling, greyscaling, frame-stacking of input images, 4 action repeatitions [sic], reward clipping at [-1, 1], and a discount factor of 0.99" |
| exploration | epsilon-greedy, epsilon = 0.01 |
| minibatch | 32 |
| replay | capacity 1M transitions; no sampling until at least 20% full; then sample "once every 4 agent steps (i.e. 16 environment frames)" |
| optimizer | Adam, step size 0.0001 |
| target network | updated every 2500 agent steps (i.e. 10,000 frames), when used |
| networks | two conv layers (kernel 8x8 then 4x4, stride 4 then 2), one fully connected hidden layer, linear output (one per action), ReLU after hidden layers |
| channels (1) / channels (2) / hidden units | small 4/8/64; medium 8/16/128; large 16/32/256; extra-large 32/64/512 |
| stability metric | maximal |Q|; soft divergence if > 100 |
| performance metric | median human-normalised score across 57 games |
Derived by us (not printed): 336 = 4 targets x 3 n values x 7 replay settings (alpha = 0, plus alpha in {1/2, 1, 2}
each with beta in {0, 0.4}) x 4 sizes. Total runs 336 x 57 x 3 = 57,456 (not stated in the paper).
Note: the study's networks have two conv layers, not the three of Nature DQN.

## 8. Results (with locations)
- Unbounded divergence (Sec. 4.1, p.5): "soft divergences do occur (values exceed 100), and sometimes grow to wildly
  unrealistic value estimates. But, surprisingly, they never became unbounded (more precisely, they never produced
  floating point NaNs). The lack of unbounded divergence in any of the runs suggests that although the deadly triad
  can cause unbounded divergence in deep RL, it is not common when applying deep Q-learning and its variants to
  current deep function approximators. This provides support for Hypothesis 1."
- Bootstrap targets (Sec. 4.2, p.6; Figure 2 left): "Q-learning exhibits by far the largest fraction of instabilities
  (61%); target Q-learning and double Q-learning, which both reduce the issue of inappropriate generalization via the
  use of target networks, are the most stable. Inverse double Q-learning ... exhibits a moderate rate (33%) of
  soft-diverging runs." Figure 2 left prints the four rates as labels: Q 61%, Target Q 14%, Inverse Double Q 33%,
  Double Q 10% (14% appears only as this printed label). "These results provide strong support for both Hypothesis 2
  and 3: divergence occurs more easily with overestimation biases, and when bootstrapping on the same network."
- Recovery (p.6): "while value estimates frequently grew quickly to over a million, they typically reduced down again
  below the threshold of 100." Figure 2 right: percentiles of max |Q| over time for two network sizes, log scale.
- Multi-step (Sec. 4.3, p.7; Figure 3a): "One-step Q-learning (n = 1) exhibits soft divergences on 94% of the runs.
  With n = 10, this reduced to 21%. ... This provides strong support for Hypothesis 4."
- Capacity (p.7; Figure 3b): "The results ran counter to Hypothesis 5. For Q-learning: small networks exhibited less
  instabilities (53%) than larger network (67%). The trend is less clear for more stable update rules; double
  Q-learning has a consistent soft-divergence rate of about 10% across network capacities."
- Prioritization (p.7; Figure 3c): "For Q-learning the rate increased from 52% to 77%. For the more stable double
  Q-learning update the rate grows from 2% to 23%." "Without importance sampling correction (marked 'UP' in Figure
  3c), the fraction of diverging runs is up to 10% higher than with importance sampling correction ('P'). These
  results provide strong support for Hypothesis 6."
- Performance (Sec. 4.4, p.7; Figure 4): "soft-divergence (unrealistic value estimation) correlates with poor control
  performance." "Some Q-learning runs exhibit both unrealistic values (e.g., around 1,000) as well as reasonably good
  performance" (overestimation with the action ranking roughly kept). n = 10: "fewer instabilities and better
  performance". Best performers use large and extra-large networks, which "also have the highest number of
  unrealistically high value estimates, at least for Q-learning." "Prioritising too heavily (alpha = 1 or alpha = 2)
  correlates both with unreasonably high value estimates as well as reduced performance."
- Target networks in the toy (Sec. 3, p.4): "target networks do not suffice as a solution to the deadly triad. When
  such target networks are applied to the standard Tsitsiklis and Van Roy example (Figure 1a) with linear function
  approximation, the weights still diverge, though the divergence is slowed down by the copying period."
- Best configuration (Table 1 caption, p.13): "10-step return, large network, Q-learning bootstrapping and uniform
  replay" (after 20M frames).
- Discussion (Sec. 5, p.8): "Action value estimates can commonly exhibit exponential initial growth, and still
  subsequently recover to plausible magnitudes." "A key result is that the instabilities caused by the deadly triad
  interact with statistical estimation issues induced by the bootstrap method used. As a result, the instabilities
  commonly observed in the standard epsilon-greedy regime of deep Q-learning can be greatly reduced by bootstrapping
  on a separate network and by reducing the overestimation bias." "These alternatives to the basic Q-learning updates
  do not, however, fully resolve the issues caused by the deadly triad."
- What would have falsified the claims: any NaN run (H1); soft-divergence rates not falling with n (H4); not rising
  with alpha (H6). H5 was in fact contradicted.

## 9. Limitations stated
- p.5: run length "is not enough for state-of-the-art performance but it is sufficient to investigate the learning
  dynamics".
- p.8: "the interaction of network capacity with the other components of the triad is more nuanced than we had
  initially hypothesized"; "the general learning dynamics and interactions between (soft)-divergence and control
  performance could benefit from further study."
- Soft divergence is a proxy: a run with |q| > 100 can still perform reasonably (p.7).

## 10. Outline feedback (ch-approx, ch-dqn)
- "what the 2018 study measured": 57 games, 336 settings per game, 3 seeds, ~20M frames; metric max |Q| > 100 = soft
  divergence. Findings in one line each: soft divergence is common (61% of Q-learning runs), unbounded divergence
  never occurred (no NaNs); target networks and double Q cut it; n = 10 cuts Q-learning from 94% to 21%; heavier
  prioritization raises it (52% to 77% for Q-learning); bigger networks diverge more for Q-learning but score better.
- "values spiral to infinity" is the linear-example behavior. In the deep study, values often blew up "to over a
  million" and then came back below 100. Say both, so the reader does not expect NaNs from DQN.
- "In practice" list: replace with source-backed items: target network (helps; "do not suffice" alone), double
  Q-learning, multi-step returns (strongest effect), avoid heavy prioritization or keep the IS correction. "Lower the
  learning rate" does not fix the textbook divergence (S&B p.260, p.262) and "clip gradients" is not in group C
  sources; label them as general advice if kept.
- viz-triad optional toggle: the w(phi + u) variant (Figure 1c) shows "diverge, then recover" with one extra
  parameter. It previews soft divergence.
- ch-dqn (group D): the inverse double Q-learning target is a clean way to separate "target network" from "double
  estimator" effects if the DQN chapter wants a citation for why both help.
- Notation is S&B's (R_{t+1}); no index shift to flag for this paper.
