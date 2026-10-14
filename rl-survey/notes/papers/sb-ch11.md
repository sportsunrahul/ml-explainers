# S&B ch. 11: *Off-policy Methods with Approximation

## 1. Identity
- Sutton and Barto, 2nd ed. (2018), RLbook2020.pdf (2020 printing). Book page = PDF page - 22. Chapter 11 starts
  p.257 (PDF 279). The chapter title carries an asterisk (advanced material).
- Sections (book pages, TOC): 11.1 Semi-gradient Methods 258; 11.2 Examples of Off-policy Divergence 260; 11.3 The
  Deadly Triad 264; 11.4 Linear Value-function Geometry 266; 11.5 Gradient Descent in the Bellman Error 269; 11.6 The
  Bellman Error is Not Learnable 274; 11.7 Gradient-TD Methods 278; 11.8 Emphatic-TD Methods 281; 11.9 Reducing
  Variance 283; 11.10 Summary 284. Bib remarks p.285-286 (PDF 307-308).
- Used for ch-approx: 11.1-11.3, plus Example 11.1 (p.263) and Sec. 11.5 Eq. 11.23 / Example 11.2 (for the "full
  gradient" exercise).

## 2. Core problem
Off-policy learning has two parts: the update target (fixed by importance sampling, as in the tabular case) and the
distribution of updates, which no longer matches the on-policy distribution. "The on-policy distribution is
important to the stability of semi-gradient methods" (p.258). Semi-gradient methods can diverge off-policy.

## 3. What prior methods did and why that hurt
- Tabular off-policy methods (ch. 5-7) extend "readily" to semi-gradient form, but "these methods may diverge in some
  cases, and in that sense are not sound, but still they are often successfully used" (p.258). They "are guaranteed
  stable and asymptotically unbiased for the tabular case" (p.258).
- Even full expected (DP-style) semi-gradient updates diverge on Baird's counterexample (p.262), and even least-squares
  refits each step diverge on Tsitsiklis and Van Roy's counterexample (Example 11.1, p.263).

## 4. The one key idea
Instability needs all three of function approximation, bootstrapping and off-policy training: the deadly triad
(Sec. 11.3). The two-state w, 2w example shows the mechanism in one line of algebra.

## 5. Mechanism, step by step
w, 2w example (Sec. 11.2, p.260, PDF 282):
1. Two states with estimated values w and 2w (features 1 and 2, one weight w).
2. From the first state, one action; it moves deterministically to the second state with reward 0.
3. TD error on that transition: (2 gamma - 1) w_t. The update multiplies w by 1 + alpha(2 gamma - 1).
4. If only this transition is ever updated (possible off-policy: rho_t = 0 on other transitions), w is multiplied by
   a constant greater than 1 at each update whenever gamma > 0.5: w goes to +-infinity.
5. On-policy, every transition out of the 2w state must also be updated, and it pulls w down (p.261).

Baird's counterexample (p.261-262): see settings table (Sec. 7). Semi-gradient off-policy TD (Eq. 11.2) and the
expected semi-gradient DP update (Eq. 11.9) both diverge.

## 6. Equations (verified on the PDF page images)
- Eq. 11.1, Sec. 11.1, p.258 (PDF 280): `\rho_t \doteq \rho_{t:t} = \frac{\pi(A_t|S_t)}{b(A_t|S_t)}` (per-step
  importance sampling ratio; pi = target policy, b = behavior policy).
- Eq. 11.2, p.258: `\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\rho_t\delta_t\nabla\hat v(S_t,\mathbf{w}_t)`.
  Source: semi-gradient off-policy TD(0) "is just like the corresponding on-policy algorithm (page 203) except for the
  addition of rho_t".
- Eq. 11.3, p.258 (episodic and discounted):
  `\delta_t \doteq R_{t+1} + \gamma\hat v(S_{t+1},\mathbf{w}_t) - \hat v(S_t,\mathbf{w}_t)`.
  Eq. 11.4 is the average-reward version: `\delta_t \doteq R_{t+1} - \bar R_t + \hat v(S_{t+1},\mathbf{w}_t) - \hat v(S_t,\mathbf{w}_t)`.
- Eq. 11.5, p.259: semi-gradient Expected Sarsa, `\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\delta_t\nabla\hat q(S_t,A_t,\mathbf{w}_t)`
  with an expected target over pi(a|S_{t+1}). "Note that this algorithm does not use importance sampling." (p.259)
- w, 2w TD error and update, Sec. 11.2, p.260 (PDF 282), UNNUMBERED (derived in the text from Eq. 11.2):
  `\delta_t = R_{t+1} + \gamma\hat v(S_{t+1},\mathbf{w}_t) - \hat v(S_t,\mathbf{w}_t) = 0 + \gamma 2w_t - w_t = (2\gamma - 1)w_t`
  `w_{t+1} = w_t + \alpha\rho_t\delta_t\nabla\hat v(S_t,w_t) = w_t + \alpha\cdot 1\cdot(2\gamma - 1)w_t\cdot 1 = \big(1 + \alpha(2\gamma - 1)\big)w_t`
  Quote (p.260): "Note that the importance sampling ratio, rho_t, is 1 on this transition because there is only one
  action available from the first state, so its probabilities of being taken under the target and behavior policies
  must both be 1. In the final update above, the new parameter is the old parameter times a scalar constant,
  1 + alpha(2 gamma - 1). If this constant is greater than 1, then the system is unstable and w will go to positive
  or negative infinity depending on its initial value. Here this constant is greater than 1 whenever gamma > 0.5.
  Note that stability does not depend on the specific step size, as long as alpha > 0. Smaller or larger step sizes
  would affect the rate at which w goes to infinity, but not whether it goes there or not."
- Eq. 11.9, p.262 (PDF 284), expected (DP-style) semi-gradient update over all states:
  `\mathbf{w}_{k+1} \doteq \mathbf{w}_k + \frac{\alpha}{|\mathcal{S}|}\sum_s\Big(\mathbb{E}_\pi[R_{t+1} + \gamma\hat v(S_{t+1},\mathbf{w}_k) \mid S_t = s] - \hat v(s,\mathbf{w}_k)\Big)\nabla\hat v(s,\mathbf{w}_k)`
- Eq. 11.10, Example 11.1 (Tsitsiklis and Van Roy's counterexample), p.263 (PDF 285): the w, 2w pair plus a terminal
  state; the 2w state loops to itself with probability 1 - epsilon and terminates with probability epsilon; reward 0
  everywhere. Least-squares refit each step:
  `w_{k+1} = \arg\min_{w\in\mathbb{R}}\sum_{s\in\mathcal{S}}\Big(\hat v(s,w) - \mathbb{E}_\pi\big[R_{t+1} + \gamma\hat v(S_{t+1},w_k) \mid S_t = s\big]\Big)^2`
  `= \arg\min_{w\in\mathbb{R}}\big(w - \gamma 2w_k\big)^2 + \big(2w - (1-\varepsilon)\gamma 2w_k\big)^2 = \frac{6 - 4\varepsilon}{5}\gamma w_k.` (11.10)
  "The sequence {w_k} diverges when gamma > 5/(6 - 4 epsilon) and w_0 != 0." (p.263)
- Eq. 11.23, Sec. 11.5, p.270 (PDF 292), the "full gradient" of the squared TD error (naive residual-gradient):
  `\mathbf{w}_{t+1} = \mathbf{w}_t - \tfrac{1}{2}\alpha\nabla(\rho_t\delta_t^2) = \mathbf{w}_t - \alpha\rho_t\delta_t\nabla\delta_t = \mathbf{w}_t + \alpha\rho_t\delta_t\big(\nabla\hat v(S_t,\mathbf{w}_t) - \gamma\nabla\hat v(S_{t+1},\mathbf{w}_t)\big)`
  "which you will recognize as the same as the semi-gradient TD algorithm (11.2) except for the additional final
  term. This term completes the gradient and makes this a true SGD algorithm with excellent convergence guarantees."
  (p.270) "Although the naive residual-gradient algorithm converges robustly, it does not necessarily converge to a
  desirable place." (p.271)

## 7. Settings
w, 2w example (Sec. 11.2, p.260):
| item | value (source wording) |
|---|---|
| states | two, "perhaps as part of a larger MDP"; values w and 2w |
| features | "single-component vectors), in this case 1 and 2" |
| weights | one component w |
| action, transition | "only one action is available, and it results deterministically in a transition to the second state with a reward of 0" |
| rho_t | 1 on this transition |
| divergence condition | gamma > 0.5 (any alpha > 0) |
| worked numbers (p.260) | "Suppose initially w = 10"; "If gamma is nearly 1, then the TD error will be nearly 10, and, if alpha = 0.1, then w will be increased to nearly 11"; second state "to nearly 22"; next TD error "~11"; next value "~12.1" |

Baird's counterexample (Sec. 11.2, p.261-262, Figures 11.1 and 11.2):
| item | value |
|---|---|
| MDP | "the episodic seven-state, two-action MDP shown in Figure 11.1" (p.261) |
| dashed action | "takes the system to one of the six upper states with equal probability" |
| solid action | "takes the system to the seventh state" |
| behavior b | dashed 6/7, solid 1/7; next-state distribution under b is uniform, "which is also the starting distribution for each episode" |
| target pi | "always takes the solid action", pi(solid|.) = 1; on-policy distribution "concentrated in the seventh state" |
| reward, gamma | "The reward is zero on all transitions. The discount rate is gamma = 0.99." |
| values | upper states i = 1..6: 2w_i + w_8; seventh state: w_7 + 2w_8 (Figure 11.1) |
| features | w in R^8; x(1) = (2, 0, 0, 0, 0, 0, 0, 1)^T (p.262) |
| true values | v_pi(s) = 0 for all s; exactly representable with w = 0; 8 weights > 7 nonterminal states; features linearly independent |
| Figure 11.2 settings | "The step size was alpha = 0.01, and the initial weights were w = (1, 1, 1, 1, 1, 1, 10, 1)^T." (caption, p.262) |
| Figure 11.2 axes | left "Semi-gradient Off-policy TD", x-axis "Steps" 0 to 1000; right "Semi-gradient DP", x-axis "Sweeps" 0 to 1000 (printed tick labels) |
Figure 11.1 caption wording: "The solid action usually results in the seventh state, and the dashed action usually
results in one of the other six states, each with equal probability. The reward is always zero." The text does not
say how episodes end.
Derived by us (not printed): from Eq. 11.1, rho = pi/b = 1/(1/7) = 7 on solid actions and 0 on dashed actions.

## 8. Findings worth remembering
- Baird (p.262): "If we apply semi-gradient TD(0) to this problem (11.2), then the weights diverge to infinity, as
  shown in Figure 11.2 (left). The instability occurs for any positive step size, no matter how small. In fact, it
  even occurs if an expected update is done as in dynamic programming (DP), as shown in Figure 11.2 (right)."
  "In this case, there is no randomness and no asynchrony, just as in a classical DP update. ... Yet still the system
  is unstable." Qualitative only (no numbers from the plot): w_8 and w_1-w_6 grow; w_7 stays low.
- Baird with on-policy distribution (p.262): "If we alter just the distribution of DP updates in Baird's
  counterexample, from the uniform distribution to the on-policy distribution (which generally requires asynchronous
  updating), then convergence is guaranteed to a solution with error bounded by (9.14)."
- p.263: "The example shows that even the simplest combination of bootstrapping and function approximation can be
  unstable if the updates are not done according to the on-policy distribution."
- Q-learning (p.263): "There are also counterexamples similar to Baird's showing divergence for Q-learning." With an
  epsilon-greedy behavior policy: "To the best of our knowledge, Q-learning has never been found to diverge in this
  case, but there has been no theoretical analysis."
- On-policy intuition (p.261): "Under on-policy training, however, rho_t is always one. Each time there is a
  transition from the w state to the 2w state, increasing w, there would also have to be a transition out of the 2w
  state." "Eventually the piper must be paid. In the on-policy case the promise of future reward must be kept and the
  system is kept in check. But in the off-policy case, a promise can be made and then, after taking an action that
  the target policy never would, forgotten and forgiven."
- Averagers (nearest neighbor, locally weighted regression) are stable; "but not popular methods such as tile coding
  and artificial neural networks (ANNs)" (p.264).
- Deadly triad, Sec. 11.3, p.264 (PDF 286). The three elements, verbatim headings and definitions:
  - "Function approximation A powerful, scalable way of generalizing from a state space much larger than the memory
    and computational resources (e.g., linear function approximation or ANNs)."
  - "Bootstrapping Update targets that include existing estimates (as in dynamic programming or TD methods) rather
    than relying exclusively on actual rewards and complete returns (as in MC methods)."
  - "Off-policy training Training on a distribution of transitions other than that produced by the target policy.
    Sweeping through the state space and updating all states uniformly, as in dynamic programming, does not respect
    the target policy and is an example of off-policy training."
  - Not control or GPI (quote): "In particular, note that the danger is not due to control or to generalized policy
    iteration. Those cases are more complex to analyze, but the instability arises in the simpler prediction case
    whenever it includes all three elements of the deadly triad. The danger is also not due to learning or to
    uncertainties about the environment, because it occurs just as strongly in planning methods, such as dynamic
    programming, in which the environment is completely known."
  - Any two (quote): "If any two elements of the deadly triad are present, but not all three, then instability can be
    avoided."
  - Which to give up (p.264-265): function approximation "most clearly cannot be given up"; giving up bootstrapping
    "is possible, at the cost of computational and data efficiency"; off-policy: "On-policy methods are often
    adequate. For model-free reinforcement learning, one can simply use Sarsa rather than Q-learning." But off-policy
    learning "is essential" for learning many predictions in parallel (p.265).
- Example 11.2 A-split (p.271): tabular, on-policy, gamma = 1; true values A = 1/2, B = 1, C = 0; the naive
  residual-gradient algorithm converges to B = 3/4, C = 1/4 (A = 1/2), "the values that minimize the TDE".
- History (bib remarks p.285, PDF 307): "The first semi-gradient method was linear TD(lambda) (Sutton, 1988). The name
  'semi-gradient' is more recent (Sutton, 2015a)." "The earliest w-to-2w example was given by Tsitsiklis and Van Roy
  (1996)". "Baird's counterexample is due to Baird (1995), though the version we present here is slightly modified."
  "The deadly triad was first identified by Sutton (1995b) and thoroughly analyzed by Tsitsiklis and Van Roy (1997).
  The name 'deadly triad' is due to Sutton (2015a)."
- Reference entries (S&B ref list): Baird (1995), "Residual algorithms: Reinforcement learning with function
  approximation", ICML 12, pp. 30-37 (p.482); Tsitsiklis and Van Roy (1997), IEEE Trans. Automatic Control
  42(5):674-690 (p.515).

## 9. Limitations stated
- Fixes for the off-policy distribution problem are open: "This is a cutting-edge research area, and it is not clear
  which of these approaches is most effective in practice." (p.258) Summary: "The whole area of off-policy learning is
  relatively new and unsettled." (p.285)
- Gradient-TD (Sec. 11.7) and Emphatic-TD (Sec. 11.8) are the book's stable alternatives; not needed for ch-approx.

## 10. Outline feedback (ch-approx)
- "w grows each step when gamma > 0.5": correct (p.260). The page should also say the book's point that the step size
  does not decide whether it diverges, only how fast (quote above). For viz-triad this means the alpha slider changes
  the slope of log|w| over steps, not the sign of the trend, in "Off-policy updates" mode with the TD target.
- "On-policy updates" button: the S&B w, 2w fragment has no transition out of the 2w state, so it cannot be simulated
  on-policy as printed. Use the complete MDP of Example 11.1 (2w state loops to itself with probability 1 - epsilon,
  terminates with probability epsilon; the 2018 paper's Figure 1a draws the same MDP). Epsilon is not given a value
  in either source: the page must choose one and say so. Derived by us (label as our derivation if shown): with
  on-policy weighting (one s1 update and 1/epsilon s2 updates per episode in expectation), the expected TD update is
  alpha w [(2 gamma - 1) + 4(gamma(1 - epsilon) - 1)/epsilon], which is negative for all gamma <= 1, so w decays.
  With uniform weighting (each state once, as Eq. 11.9) it is (alpha/2) w [gamma(6 - 4 epsilon) - 5], which diverges
  exactly when gamma > 5/(6 - 4 epsilon), the same threshold as Eq. 11.10.
- "Monte Carlo target" button: all rewards are 0, so the MC target is 0 and the update is w <- w(1 - alpha phi^2)
  (derived; phi = 1 or 2). It converges for small alpha in any update mode. Unlike the TD case, alpha does matter
  here (too large an alpha overshoots), so state the alpha range the viz allows.
- Optional toggle (2018 paper, Sec. 2, p.3): v(s) = w(phi(s) + u) with u learnable "first seemingly diverge, then
  recover and converge". It previews the deep-RL finding (soft divergence that recovers).
- "The triad named; any two are safe": the book says "instability can be avoided" with any two (p.264), not that any
  two are always safe. Use the book's wording. Also add the book's point that the danger is "not due to control or
  to generalized policy iteration" and occurs in planning (DP) too: it answers "is this a Q-learning bug?".
- Self-check "which two triad ingredients does tabular Q-learning have": supported. Tabular Q-learning bootstraps and
  is off-policy, and tabular is a special case of linear function approximation with no generalization. S&B p.258:
  these methods "are guaranteed stable and asymptotically unbiased for the tabular case". The 2018 paper p.5: "this is
  why the tabular version of off-policy TD does not diverge."
- Baird paragraph: settings are complete in the table above. Note the text calls it "episodic" but gives no
  termination rule; Figure 11.1 shows none. If the page simulates it, follow the figure (solid -> state 7, dashed ->
  uniform over the six upper states) and say so.
- PyTorch exercise "full gradient version (no detach)": this is the naive residual-gradient algorithm, Eq. 11.23
  (Sec. 11.5). On a stochastic task such as the random walk, it converges robustly but to the minimizer of the mean
  squared TD error, not to the true values (Example 11.2, A-split). So "print both errors" should show the full
  gradient version settling at a worse RMS error. Cite Eq. 11.23 and Example 11.2 in the exercise text.
- "In practice: lower the learning rate ..." conflicts with the sources for true triad divergence: in the w, 2w
  example (p.260) and Baird (p.262) a smaller alpha only slows divergence. Keep "lower the learning rate" only as a
  generic tip for noisy training, not as a triad fix. See deadlytriad2018.md for the measured fixes.
