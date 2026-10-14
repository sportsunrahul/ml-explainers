# S&B ch. 9: On-policy Prediction with Approximation

## 1. Identity
- Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), file RLbook2020.pdf (2020 printing).
  Book page = PDF page - 22. Chapter 9 starts p.197 (PDF 219).
- Sections (book pages, from the TOC): 9.1 Value-function Approximation 198; 9.2 The Prediction Objective (VE) 199;
  9.3 Stochastic-gradient and Semi-gradient Methods 200; 9.4 Linear Methods 204; 9.5 Feature Construction for Linear
  Methods 210 (9.5.5 Radial Basis Functions p.221); 9.6 Selecting Step-Size Parameters Manually 222; 9.7 Nonlinear
  Function Approximation: Artificial Neural Networks 223; 9.8 Least-Squares TD 228; 9.9 Memory-based 230; 9.10
  Kernel-based 232; 9.11 Interest and Emphasis 234; 9.12 Summary 236. Bib remarks p.237-238.
- Used for ch-approx: 9.1-9.4 (and 9.5.5 for the radial features of viz-generalize).

## 2. Core problem
A table needs one entry per state. With far more states than weights (d << |S|), one weight change moves many
states' values. So we must say which states' errors matter (a weighting mu), and we need an update that learns
online from nonstationary (bootstrapped) targets.

## 3. What prior methods did and why that hurt (per the source)
- Tabular updates moved one entry toward the target u and left the others unchanged (p.198). That does not scale and
  does not generalize.
- Off-the-shelf supervised methods "assume a static training set over which multiple passes are made" (p.198). RL
  needs online learning and "nonstationary target functions". Bootstrapped targets are nonstationary even for a fixed
  policy (p.198).

## 4. The one key idea
Treat each update s -> u as a supervised training example and do SGD on the squared error. With a bootstrapped
target, differentiate only the estimate, not the target: a "semi-gradient" method.

## 5. Mechanism, step by step
1. Pick the objective VE (Eq. 9.1), weighted by the on-policy distribution mu (fraction of time spent in s).
2. With the true value as target, SGD on the per-example squared error gives Eq. 9.4 -> 9.5.
3. Replace v_pi(S_t) by a target U_t (Eq. 9.7). If U_t is unbiased (Monte Carlo G_t), this is true SGD and converges
   to a local optimum under the step-size conditions (2.7) (p.202).
4. With a bootstrapped target, the step 9.4 -> 9.5 is invalid, so the update is only part of the gradient (quote
   below). Semi-gradient TD(0) uses U_t = R_{t+1} + gamma v-hat(S_{t+1}, w) (p.203).
5. Linear case: v-hat = w^T x(s) and the gradient is x(s). The TD update (Eq. 9.9) has the expected form
   w + alpha(b - A w) (Eq. 9.10). If it converges, it converges to the TD fixed point w_TD = A^{-1} b (Eq. 9.12).
6. On-policy, continuing, gamma < 1: A = X^T D (I - gamma P) X is positive definite, so linear TD(0) is stable
   (box p.206-207). The VE at the fixed point is at most 1/(1 - gamma) times the minimum (Eq. 9.14).

Gradient Monte Carlo box (p.202, PDF 224), transcribed:
```
Gradient Monte Carlo Algorithm for Estimating v-hat ~ v_pi
Input: the policy pi to be evaluated
Input: a differentiable function v-hat : S x R^d -> R
Algorithm parameter: step size alpha > 0
Initialize value-function weights w in R^d arbitrarily (e.g., w = 0)
Loop forever (for each episode):
  Generate an episode S_0, A_0, R_1, S_1, A_1, ..., R_T, S_T using pi
  Loop for each step of episode, t = 0, 1, ..., T-1:
    w <- w + alpha [G_t - v-hat(S_t,w)] grad v-hat(S_t,w)
```

Semi-gradient TD(0) box (p.203, PDF 225), transcribed:
```
Semi-gradient TD(0) for estimating v-hat ~ v_pi
Input: the policy pi to be evaluated
Input: a differentiable function v-hat : S+ x R^d -> R such that v-hat(terminal, .) = 0
Algorithm parameter: step size alpha > 0
Initialize value-function weights w in R^d arbitrarily (e.g., w = 0)
Loop for each episode:
  Initialize S
  Loop for each step of episode:
    Choose A ~ pi(.|S)
    Take action A, observe R, S'
    w <- w + alpha [R + gamma v-hat(S',w) - v-hat(S,w)] grad v-hat(S,w)
    S <- S'
  until S is terminal
```
What is held constant: the target R + gamma v-hat(S',w) is treated as a constant (no gradient through v-hat(S',w)).
Only v-hat(S,w) is differentiated. In autograd terms, the target is detached.

## 6. Equations (verified on the PDF page images)
- Eq. 9.1, Sec. 9.2, p.199 (PDF 221). Mean square value error:
  `\overline{VE}(\mathbf{w}) \doteq \sum_{s\in\mathcal{S}} \mu(s)\Big[v_\pi(s) - \hat v(s,\mathbf{w})\Big]^2`
  Gloss: mu(s) >= 0 with sum_s mu(s) = 1 says how much we care about state s; v_pi is the true value; v-hat(s,w) is the
  estimate. The root VE "gives a rough measure of how much the approximate values differ from the true values and is
  often used in plots" (p.199). On-policy distribution: mu = fraction of time spent in s; in continuing tasks, the
  stationary distribution under pi. Episodic version: Eqs. 9.2-9.3 (box p.199).
- Eq. 9.4 and Eq. 9.5, Sec. 9.3, p.201 (PDF 223):
  `\mathbf{w}_{t+1} \doteq \mathbf{w}_t - \tfrac{1}{2}\alpha\nabla\Big[v_\pi(S_t) - \hat v(S_t,\mathbf{w}_t)\Big]^2` (9.4)
  `= \mathbf{w}_t + \alpha\Big[v_\pi(S_t) - \hat v(S_t,\mathbf{w}_t)\Big]\nabla\hat v(S_t,\mathbf{w}_t)` (9.5)
  Gloss: alpha is a positive step size; grad is the column vector of partials w.r.t. w (Eq. 9.6). Source: "the
  overall step in w_t is proportional to the negative gradient of the example's squared error (9.4)" (p.201).
- Eq. 9.7, p.202 (PDF 224), general SGD with target U_t:
  `\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\Big[U_t - \hat v(S_t,\mathbf{w}_t)\Big]\nabla\hat v(S_t,\mathbf{w}_t)`
  "If U_t is an unbiased estimate, that is, if E[U_t|S_t=s] = v_pi(s), for each t, then w_t is guaranteed to converge
  to a local optimum under the usual stochastic approximation conditions (2.7) for decreasing alpha." (p.202)
- Semi-gradient TD(0) update: NOT numbered in Sec. 9.3. It is Eq. 9.7 with
  `U_t \doteq R_{t+1} + \gamma\hat v(S_{t+1},\mathbf{w})` (p.203) and appears only in the box (p.203). Its linear
  form is Eq. 9.9. Its off-policy form is Eq. 11.2 with delta_t from Eq. 11.3.
- Why "semi-gradient" (quote, p.202, PDF 224): "One way to look at this is that the key step from (9.4) to (9.5)
  relies on the target being independent of w_t. This step would not be valid if a bootstrapping estimate were used
  in place of v_pi(S_t). Bootstrapping methods are not in fact instances of true gradient descent (Barnard, 1993).
  They take into account the effect of changing the weight vector w_t on the estimate, but ignore its effect on the
  target. They include only a part of the gradient and, accordingly, we call them semi-gradient methods."
- Eq. 9.8, Sec. 9.4, p.205 (PDF 227): `\hat v(s,\mathbf{w}) \doteq \mathbf{w}^\top\mathbf{x}(s) \doteq \sum_{i=1}^d w_i x_i(s)`.
  Unnumbered on p.205: `\nabla\hat v(s,\mathbf{w}) = \mathbf{x}(s)` and the linear SGD update
  `\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\Big[U_t - \hat v(S_t,\mathbf{w}_t)\Big]\mathbf{x}(S_t)`.
- Eq. 9.9, p.205 (linear semi-gradient TD(0), continuing case):
  `\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\Big(R_{t+1} + \gamma\mathbf{w}_t^\top\mathbf{x}_{t+1} - \mathbf{w}_t^\top\mathbf{x}_t\Big)\mathbf{x}_t`
  `= \mathbf{w}_t + \alpha\Big(R_{t+1}\mathbf{x}_t - \mathbf{x}_t\big(\mathbf{x}_t - \gamma\mathbf{x}_{t+1}\big)^\top\mathbf{w}_t\Big)`
  with x_t = x(S_t).
- Eq. 9.10, p.205: `\mathbb{E}[\mathbf{w}_{t+1}|\mathbf{w}_t] = \mathbf{w}_t + \alpha(\mathbf{b} - \mathbf{A}\mathbf{w}_t)`
- Eq. 9.11, p.206 (PDF 228): `\mathbf{b} \doteq \mathbb{E}[R_{t+1}\mathbf{x}_t] \in \mathbb{R}^d` and
  `\mathbf{A} \doteq \mathbb{E}\Big[\mathbf{x}_t\big(\mathbf{x}_t - \gamma\mathbf{x}_{t+1}\big)^\top\Big] \in \mathbb{R}^{d\times d}`
- Eq. 9.12, p.206: `\mathbf{b} - \mathbf{A}\mathbf{w}_{TD} = \mathbf{0} \Rightarrow \mathbf{b} = \mathbf{A}\mathbf{w}_{TD} \Rightarrow \mathbf{w}_{TD} \doteq \mathbf{A}^{-1}\mathbf{b}`.
  "This quantity is called the TD fixed point. In fact linear semi-gradient TD(0) converges to this point." (p.206)
- Eq. 9.13 (box p.206): `\mathbb{E}[\mathbf{w}_{t+1}|\mathbf{w}_t] = (\mathbf{I} - \alpha\mathbf{A})\mathbf{w}_t + \alpha\mathbf{b}`.
  Box: if a diagonal element of A is negative, the matching element of I - alpha A exceeds one, and that component is
  amplified, "which will lead to divergence if continued". A positive definite -> stable. Unnumbered in the box:
  `\mathbf{A} = \mathbf{X}^\top\mathbf{D}(\mathbf{I} - \gamma\mathbf{P})\mathbf{X}`. The column sums of D(I - gamma P)
  are (1 - gamma) mu^T > 0 because mu is stationary. "Thus, the key matrix and its A matrix are positive definite,
  and on-policy TD(0) is stable." (p.207)
- Eq. 9.14, p.207 (PDF 229): `\overline{VE}(\mathbf{w}_{TD}) \le \frac{1}{1-\gamma}\min_{\mathbf{w}}\overline{VE}(\mathbf{w})`
  Source wording: "That is, the asymptotic error of the TD method is no more than 1/(1-gamma) times the smallest
  possible error, that attained in the limit by the Monte Carlo method. Because gamma is often near one, this
  expansion factor can be quite large, so there is substantial potential loss in asymptotic performance with the TD
  method." Proven "in the continuing case" (p.207).
- RBF feature, Sec. 9.5.5, p.221 (PDF 243), unnumbered:
  `x_i(s) \doteq \exp\left(-\frac{\|s - c_i\|^2}{2\sigma_i^2}\right)`; c_i is the center and sigma_i the width
  (Figure 9.13 is a 1D example).

## 7. Settings
ch-approx needs no numbered experiment from this chapter. Example 9.1 (1000-state walk, gradient MC, state
aggregation in 10 groups of 100, alpha = 2 x 10^-5, 100,000 episodes; p.203) is available if the page wants a
generalization picture. Figure 9.2 data are "averages over 100 runs" (p.208).

## 8. Findings worth remembering
- p.208 (PDF 230): "Critical to these convergence results is that states are updated according to the on-policy
  distribution. For other update distributions, bootstrapping methods using function approximation may actually
  diverge to infinity." (This is the bridge to ch. 11.)
- p.202-203: semi-gradient methods "do not converge as robustly as gradient methods" but "converge reliably in
  important cases such as the linear case". They "typically enable significantly faster learning", and they let
  learning be "continual and online".
- p.205: the linear case has one optimum, so gradient MC converges to the global optimum of VE (with decreasing
  alpha). Semi-gradient TD(0) converges to a point "near the local optimum", not to the global optimum.
- History (bib remarks p.237-238, PDF 259-260): "Semi-gradient TD(0) was first explored by Sutton (1984, 1988)".
  "The term 'semi-gradient' to describe these bootstrapping methods is new to the second edition of this book."
  Tsitsiklis and Van Roy (1997) proved the bound on the asymptotic error of linear bootstrapping methods (Eq. 9.14).
  Widrow and Hoff (1960) LMS is "the prototypical incremental gradient-descent algorithm".

## 9. Limitations stated
- VE may not be the right objective for control: "The best value function for this purpose is not necessarily the
  best for minimizing VE" (p.200).
- Nonlinear approximators: often only local optima. "for many cases of interest in reinforcement learning there is
  no guarantee of convergence to an optimum, or even to within a bounded distance of an optimum. Some methods may in
  fact diverge" (p.200).
- Eq. 9.14 holds in the continuing case. The episodic case has a "slightly different but related bound" (p.208).

## 10. Outline feedback (ch-approx)
- "semi-gradient TD(0) (S&B section 9.3 update)": Section 9.3 gives this update only in the box on p.203, with no
  equation number. Cite it as "Eq. 9.7 with U_t = R_{t+1} + gamma v-hat(S_{t+1}, w)", or use the numbered linear
  form Eq. 9.9 (Sec. 9.4). Do not print a made-up number for it.
- The "detach" framing matches the source exactly (quote above: the method ignores the effect of the weights on the
  target).
- Optional and cheap: show Eq. 9.14 (TD fixed point within 1/(1 - gamma) of the best VE) as the price of
  bootstrapping, next to the triad. It makes "TD is biased but stable on-policy" concrete.
- viz-generalize radial features: the RBF form above is in Sec. 9.5.5 (unnumbered). Cite "S&B Sec. 9.5.5".
