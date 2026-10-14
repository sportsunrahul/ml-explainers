# Outline feedback from the extraction agents (fold into outline.md and the build brief)

## Group A (S&B ch. 1-4)
- ch-dp: S&B Figure 4.1 rows k = 1, 2, 3, 10 come from two-array (synchronous) sweeps, the sequence Eq. 4.5 defines,
  not from the in-place box. Our KIT.evalSweep is synchronous: matches. The NumPy check must use two arrays or compare
  only k = infinity. S&B Exercise 4.4: the policy iteration box can loop forever when two policies tie; break ties
  deterministically and stop when the policy is stable.
- ch-dp: S&B ch. 1-4 give no factor-gamma contraction rate ("converges in the limit"); Example 4.1 has gamma = 1. The
  page must derive the bound itself, label it "derived here", and plot it on the yard.
- ch-bandits: general update form is Eq. 2.4 (incremental mean is Eq. 2.3). Ch. 2 writes the reward of A_t as R_t;
  R_{t+1} starts in ch. 3 (footnote, p.48). The book's experiments used a baseline that also included R_t (footnote 1,
  p.37). c = 2 and alpha 0.1/0.4 appear only as curve labels. Figure 2.3 uses constant alpha = 0.1, not sample averages.
- ch-score: Law of Effect: Thorndike 1911; puzzle boxes 1898 (section 14.3). J(pi) is our notation. Undiscounted return
  is Eq. 3.7.
- ch-bellman: v_pi from q_pi has no numbered equation (it is Exercise 3.12); q_pi from v_pi is Eq. 4.6. Exercise 3.14 is a
  ready-made self-check. S&B writes true values in lowercase (v, q).
- ch-mdp: 1/(1-gamma) is Eq. 3.10 (return for constant reward +1); the book never calls it a horizon. The two-chargers
  crossover gamma^4 = 0.1 ignores the -0.04 step reward: recompute with it, or set the variant's step reward to 0.

## Group B (S&B ch. 5-8, Sutton 1988, Watkins-Dayan 1992)
- "Figure 6.2" and "Figure 6.4" are wrong for this printing (6.2 = batch training, 6.4 = backup diagrams). The Example
  6.2 curves and the cliff graph have no figure number: cite the Example, not a figure.
- Cliff: the book states only epsilon = 0.1; no alpha, no run count; 4x12 only from the figure. Our alpha is our choice.
- Example 6.7: the number of B's actions is not stated (our choice, say so).
- Figure 7.2: text says only "intermediate n" worked best; 19-state true values not printed (derive them).
- Dyna maze: layout only from the figure (uncertain); the book omits episode 1 (about 1700 steps) from its curve.
- Sutton 1988 labels the walk A-G starting at D; S&B labels A-E starting at C.
- Watkins-Dayan: "episode" means one transition; their r_n index matches deep-RL r_t, not S&B R_{t+1}.
- Example 5.5 is a sourced world for viz-ratio; Example 6.4 is the book's own TD-vs-MC explanation (certainty equivalence).
- CliffWalking-v1 exists in the venv (gymnasium 1.4.0).

## Group G (bridge papers)
- GRPO: DeepSeekMath gives the advantage no equation number (inline, Sec. 4.1.2); objective Eq. 3, KL estimator Eq. 4.
  Algorithm 1's "Equation 21" is the gradient coefficient. R1 v2 numbers the advantage (Eq. 3) but its objective uses a
  per-output ratio with no 1/|o_i|. The toy must pick one form and say which.
- InstructGPT Eq. 2: gamma is the pretraining-loss weight (27.8), not a discount. Cite App. C.4 for "no discount" and
  rename gamma on the page.
- DPO v3 abstract no longer says DPO removes "fitting a reward model"; quote "avoids fitting an explicit, standalone
  reward model" (Sec. 4).
- World Models trained C inside the dream only for Doom; CarRacing 906 is from the real environment; C trained by CMA-ES.
- MuZero plans with its model (MCTS); it is not "trained in imagination". AlphaZero uses known game rules, not a
  learned model.
- DreamerV3's actor uses REINFORCE; keep it apart from Dreamer V1's model-gradient actor.
- R1-Zero 86.7%: the sources give two vote counts (16 and 64), so print no count. Final R1's 79.8% -> 86.7% is separate.
- Claim values that must appear verbatim if used: 5 x 10^6, 10^8, 6 x 10^-4, 8 x 10^-5, 4 and 9, a single update.

## Group C (S&B ch. 9-13, Williams 1992, Sutton et al. 2000, deadly triad 2018)
- Semi-gradient TD(0) has no equation number in Sec. 9.3: cite Eq. 9.7 with the TD target U_t, or the linear Eq. 9.9.
- viz-triad "On-policy" button needs the complete MDP of Example 11.1 (the w, 2w fragment has nothing after the 2w
  state). Neither source gives an epsilon. The step size changes only how fast w diverges, not whether (p.260).
- "Any two are safe": the book says "instability can be avoided" (p.264). Quote that.
- "In practice" tips: "lower the learning rate" and "clip gradients" are not triad fixes in the sources. The 2018 study
  supports target networks (partial), double Q-learning, n-step returns and lighter prioritization.
- The PyTorch "full gradient" exercise is the naive residual-gradient update (Eq. 11.23); it settles where the squared TD
  error is smallest, not at the true values (Example 11.2): expect a worse value error.
- Deep RL divergence: soft divergence (values above 100) hit 61% of Q-learning runs, but no run went unbounded (no NaNs).
  "Spiral to infinity" describes only the linear examples.
- Eq. 13.5 is "proportional to"; the constant is the average episode length. Sutton 2000 Eq. 2 is an equality.
- gamma^t factor: in all three algorithm boxes but not in Eq. 13.8; the page's code must pick one form and say so.
- Short corridor: the book says "about -11.6"; the computed optimum is -11.657. Show the book's value (or both).
- Claim value formats: negatives use ASCII hyphen (-11.6, -44, -82); powers of two as 2^-13; about 12 TeX values.

## Group F (DPG, DDPG, TD3, SAC, SAC v2)
- TD3: Eq. 10 is clipped double-Q with the current actor and no smoothing noise; the full TD3 target is Eq. 15 (and
  Algorithm 1). Use Eq. 15 for the worked example and the self-check.
- viz-dpg: in TD3 the actor climbs Q_theta1 only; the min is in the critic target. An actor that climbs the min is SAC's
  mechanism. In a one-step toy with no bootstrap, TD3's min does nothing: redesign the toy (overestimation needs
  bootstrapped targets).
- viz-sacflow matches 1812.05905v2 (no V network, target Q copies, learned alpha), not 1801.01290v2 (V_psi + target V,
  no target Q). Name the version. "log std" appears in neither paper.
- Learned temperature: 1812.05905v2 Eq. 18, target entropy -dim(A).
- Gaussian policy: S&B Sec. 13.7, Eqs. 13.18-13.20, p.335 (5 sb- claims in F.json).
- DDPG: printed Algorithm 1 has no terminal mask (TD3 App. D gives it); "DQN's tricks" step omits batch norm; mu' means
  both the target actor and the exploration policy.
- TD3 Table 1's SAC column is an older SAC run (App. E): do not present it as TD3 vs SAC.
- Silver 2014's "SAC" means stochastic actor-critic.
- All four deep RL papers write r_t or r for S&B's R_{t+1}.
