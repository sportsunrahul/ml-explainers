# S&B ch. 13: Policy Gradient Methods

## 1. Identity
- Sutton and Barto, 2nd ed. (2018), RLbook2020.pdf (2020 printing). Book page = PDF page - 22. Chapter 13 starts
  p.321 (PDF 343).
- Sections (TOC): 13.1 Policy Approximation and its Advantages 322; 13.2 The Policy Gradient Theorem 324; 13.3
  REINFORCE: Monte Carlo Policy Gradient 326; 13.4 REINFORCE with Baseline 329; 13.5 Actor-Critic Methods 331; 13.6
  Policy Gradient for Continuing Problems 333; 13.7 Policy Parameterization for Continuous Actions 335; 13.8 Summary
  337. Bib remarks p.337-338 (PDF 359-360).
- Used for ch-pg (13.1-13.4), ch-ac (13.5) and the Gaussian part of ch-continuous (13.7).
- Notation: theta in R^{d'} = policy parameters; w in R^d = value weights; pi(a|s, theta) = Pr{A_t = a | S_t = s,
  theta_t = theta} (p.321). Reward after A_t is R_{t+1}, as in the rest of the book.

## 2. Core problem
Learn a parameterized policy directly and improve it by gradient ascent on a performance measure J(theta). The
gradient seems to need the derivative of the state distribution, which depends on the unknown environment (p.324-326).

## 3. What prior methods did and why that hurt (per the source)
- Action-value methods with epsilon-greedy selection always keep an epsilon chance of a random action and cannot
  approach a deterministic policy (p.322). They also have "no natural way of finding stochastic optimal policies"
  (p.323); see Example 13.1.
- "in epsilon-greedy selection the action probabilities may change dramatically for an arbitrarily small change in
  the estimated action values, if that change results in a different action having the maximal value. Largely
  because of this, stronger convergence guarantees are available for policy-gradient methods than for action-value
  methods." (p.324)

## 4. The one key idea
The policy gradient theorem (Eq. 13.5) gives the gradient without any derivative of the state distribution. Sampling
it gives REINFORCE: return times the eligibility vector grad ln pi(A_t|S_t, theta).

## 5. Mechanism, step by step
1. Parameterize: soft-max in action preferences (Eq. 13.2), preferences linear in features (Eq. 13.3) or an ANN.
2. Performance (episodic): J(theta) = v_{pi_theta}(s_0) (Eq. 13.4); the text assumes gamma = 1, the boxes allow
   discounting (p.324).
3. Policy gradient theorem (Eq. 13.5), proved in the box on p.325.
4. Turn the sum over states into an expectation under pi (Eq. 13.6). Stopping here gives the all-actions method
   (Eq. 13.7).
5. Multiply and divide by pi(a|S_t, theta), replace a by the sample A_t, then replace q_pi(S_t, A_t) by G_t
   (unnumbered chain, p.327). Result: REINFORCE (Eq. 13.8).
6. Subtract a baseline b(S_t) that does not depend on the action (Eq. 13.10-13.11). Expected update unchanged,
   variance can drop.
7. Replace G_t by the one-step return and use v-hat as the baseline: one-step actor-critic (Eq. 13.12-13.14).

REINFORCE box (p.328, PDF 350), transcribed:
```
REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for pi_*
Input: a differentiable policy parameterization pi(a|s, theta)
Algorithm parameter: step size alpha > 0
Initialize policy parameter theta in R^{d'} (e.g., to 0)
Loop forever (for each episode):
  Generate an episode S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T, following pi(.|., theta)
  Loop for each step of the episode t = 0, 1, ..., T-1:
    G <- sum_{k=t+1}^{T} gamma^{k-t-1} R_k                    (G_t)
    theta <- theta + alpha gamma^t G grad ln pi(A_t|S_t, theta)
```
REINFORCE with Baseline box (p.330, PDF 352), transcribed:
```
REINFORCE with Baseline (episodic), for estimating pi_theta ~ pi_*
Input: a differentiable policy parameterization pi(a|s, theta)
Input: a differentiable state-value function parameterization v-hat(s, w)
Algorithm parameters: step sizes alpha^theta > 0, alpha^w > 0
Initialize policy parameter theta in R^{d'} and state-value weights w in R^d (e.g., to 0)
Loop forever (for each episode):
  Generate an episode S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T, following pi(.|., theta)
  Loop for each step of the episode t = 0, 1, ..., T-1:
    G <- sum_{k=t+1}^{T} gamma^{k-t-1} R_k                    (G_t)
    delta <- G - v-hat(S_t, w)
    w <- w + alpha^w delta grad v-hat(S_t, w)
    theta <- theta + alpha^theta gamma^t delta grad ln pi(A_t|S_t, theta)
```
One-step Actor-Critic box (p.332, PDF 354), transcribed:
```
One-step Actor-Critic (episodic), for estimating pi_theta ~ pi_*
Input: a differentiable policy parameterization pi(a|s, theta)
Input: a differentiable state-value function parameterization v-hat(s, w)
Parameters: step sizes alpha^theta > 0, alpha^w > 0
Initialize policy parameter theta in R^{d'} and state-value weights w in R^d (e.g., to 0)
Loop forever (for each episode):
  Initialize S (first state of episode)
  I <- 1
  Loop while S is not terminal (for each time step):
    A ~ pi(.|S, theta)
    Take action A, observe S', R
    delta <- R + gamma v-hat(S', w) - v-hat(S, w)     (if S' is terminal, then v-hat(S', w) = 0)
    w <- w + alpha^w delta grad v-hat(S, w)
    theta <- theta + alpha^theta I delta grad ln pi(A|S, theta)
    I <- gamma I
    S <- S'
```
What is held constant: in the boxes, delta (or G) multiplies grad ln pi and is not differentiated w.r.t. theta; the
critic is trained by its own semi-gradient update (w line). The book also gives "Actor-Critic with Eligibility
Traces" boxes (episodic p.332, continuing p.333); not needed by the page.

## 6. Equations (verified on the PDF page images)
- Eq. 13.1, p.321 (PDF 343): `\boldsymbol\theta_{t+1} = \boldsymbol\theta_t + \alpha\widehat{\nabla J(\boldsymbol\theta_t)}`
  (stochastic gradient ascent; the hat term is "a stochastic estimate whose expectation approximates the gradient").
- Eq. 13.2, Sec. 13.1, p.322 (PDF 344): `\pi(a|s,\boldsymbol\theta) \doteq \frac{e^{h(s,a,\boldsymbol\theta)}}{\sum_b e^{h(s,b,\boldsymbol\theta)}}`.
  "We call this kind of policy parameterization soft-max in action preferences." h = numerical preference.
- Eq. 13.3, p.322: `h(s,a,\boldsymbol\theta) = \boldsymbol\theta^\top\mathbf{x}(s,a)`.
- Eq. 13.4, Sec. 13.2, p.324 (PDF 346): `J(\boldsymbol\theta) \doteq v_{\pi_{\boldsymbol\theta}}(s_0)`.
- Eq. 13.5, p.326 (PDF 348), policy gradient theorem (episodic):
  `\nabla J(\boldsymbol\theta) \propto \sum_s \mu(s)\sum_a q_\pi(s,a)\nabla\pi(a|s,\boldsymbol\theta)`
  Source: "The symbol \propto here means 'proportional to'. In the episodic case, the constant of proportionality is
  the average length of an episode, and in the continuing case it is 1, so that the relationship is actually an
  equality. The distribution mu here (as in Chapters 9 and 10) is the on-policy distribution under pi". The theorem
  "does not involve the derivative of the state distribution" (p.326). Proof box (episodic) p.325; continuing proof
  p.334-335. The box's last steps: grad J = sum_s eta(s) sum_a grad pi q = (sum_{s'} eta(s')) sum_s mu(s) sum_a ...,
  so the constant is sum_s eta(s), the expected number of steps per episode (eta from Eq. 9.2).
- Eq. 13.6, Sec. 13.3, p.326: `\nabla J(\boldsymbol\theta) \propto \sum_s\mu(s)\sum_a q_\pi(s,a)\nabla\pi(a|s,\boldsymbol\theta) = \mathbb{E}_\pi\Big[\sum_a q_\pi(S_t,a)\nabla\pi(a|S_t,\boldsymbol\theta)\Big]`
- Eq. 13.7, p.326 (all-actions method): `\boldsymbol\theta_{t+1} \doteq \boldsymbol\theta_t + \alpha\sum_a\hat q(S_t,a,\mathbf{w})\nabla\pi(a|S_t,\boldsymbol\theta)`
- REINFORCE derivation, p.327 (PDF 349), UNNUMBERED chain from (13.6):
  `\nabla J(\boldsymbol\theta) \propto \mathbb{E}_\pi\Big[\sum_a\pi(a|S_t,\boldsymbol\theta)q_\pi(S_t,a)\frac{\nabla\pi(a|S_t,\boldsymbol\theta)}{\pi(a|S_t,\boldsymbol\theta)}\Big]`
  `= \mathbb{E}_\pi\Big[q_\pi(S_t,A_t)\frac{\nabla\pi(A_t|S_t,\boldsymbol\theta)}{\pi(A_t|S_t,\boldsymbol\theta)}\Big]` (replacing a by the sample A_t ~ pi)
  `= \mathbb{E}_\pi\Big[G_t\frac{\nabla\pi(A_t|S_t,\boldsymbol\theta)}{\pi(A_t|S_t,\boldsymbol\theta)}\Big]` (because E_pi[G_t|S_t, A_t] = q_pi(S_t, A_t))
- Eq. 13.8, p.327: `\boldsymbol\theta_{t+1} \doteq \boldsymbol\theta_t + \alpha G_t\frac{\nabla\pi(A_t|S_t,\boldsymbol\theta_t)}{\pi(A_t|S_t,\boldsymbol\theta_t)}`
  Source: "Each increment is proportional to the product of a return G_t and a vector, the gradient of the probability
  of taking the action actually taken divided by the probability of taking that action." Increase "proportional to
  the return, and inversely proportional to the action probability" (p.327). No gamma^t factor (text treats gamma = 1).
- Log form (p.327): the box uses "the compact expression grad ln pi(A_t|S_t, theta_t) for the fractional vector";
  identity `\nabla\ln x = \frac{\nabla x}{x}`.
- "eligibility vector" (p.328, PDF 350): "This vector has been given several names and notations in the literature;
  we will refer to it simply as the eligibility vector. Note that it is the only place that the policy
  parameterization appears in the algorithm."
- gamma^t (p.328): "The second difference between the pseudocode update and the REINFORCE update equation (13.8) is
  that the former includes a factor of gamma^t. This is because, as mentioned earlier, in the text we are treating
  the non-discounted case (gamma = 1) while in the boxed algorithms we are giving the algorithms for the general
  discounted case." Exercise 13.2 asks the reader to derive the gamma^t version.
- Eq. 13.9, Exercise 13.3, p.329 (PDF 351), eligibility vector for linear soft-max:
  `\nabla\ln\pi(a|s,\boldsymbol\theta) = \mathbf{x}(s,a) - \sum_b\pi(b|s,\boldsymbol\theta)\mathbf{x}(s,b)`
- Eq. 13.10, Sec. 13.4, p.329: `\nabla J(\boldsymbol\theta) \propto \sum_s\mu(s)\sum_a\Big(q_\pi(s,a) - b(s)\Big)\nabla\pi(a|s,\boldsymbol\theta)`
  Why no bias (unnumbered, p.329): "The baseline can be any function, even a random variable, as long as it does not
  vary with a; the equation remains valid because the subtracted quantity is zero:"
  `\sum_a b(s)\nabla\pi(a|s,\boldsymbol\theta) = b(s)\nabla\sum_a\pi(a|s,\boldsymbol\theta) = b(s)\nabla 1 = 0.`
- Eq. 13.11, p.329: `\boldsymbol\theta_{t+1} \doteq \boldsymbol\theta_t + \alpha\Big(G_t - b(S_t)\Big)\frac{\nabla\pi(A_t|S_t,\boldsymbol\theta_t)}{\pi(A_t|S_t,\boldsymbol\theta_t)}`
  "In general, the baseline leaves the expected value of the update unchanged, but it can have a large effect on its
  variance." (p.329) "for MDPs the baseline should vary with state" (p.329). Natural choice: v-hat(S_t, w).
- Step-size rule of thumb for the critic (p.330): `\alpha^{\mathbf{w}} = 0.1/\mathbb{E}\big[\|\nabla\hat v(S_t,\mathbf{w})\|_\mu^2\big]`
  (Sec. 9.6). The policy step size alpha^theta is "much less clear"; its best value "depends on the range of variation
  of the rewards and on the policy parameterization" (p.330).
- Eq. 13.12-13.14, Sec. 13.5, p.331 (PDF 353), one-step actor-critic:
  `\boldsymbol\theta_{t+1} \doteq \boldsymbol\theta_t + \alpha\Big(G_{t:t+1} - \hat v(S_t,\mathbf{w})\Big)\frac{\nabla\pi(A_t|S_t,\boldsymbol\theta_t)}{\pi(A_t|S_t,\boldsymbol\theta_t)}` (13.12)
  `= \boldsymbol\theta_t + \alpha\Big(R_{t+1} + \gamma\hat v(S_{t+1},\mathbf{w}) - \hat v(S_t,\mathbf{w})\Big)\frac{\nabla\pi(A_t|S_t,\boldsymbol\theta_t)}{\pi(A_t|S_t,\boldsymbol\theta_t)}` (13.13)
  `= \boldsymbol\theta_t + \alpha\delta_t\frac{\nabla\pi(A_t|S_t,\boldsymbol\theta_t)}{\pi(A_t|S_t,\boldsymbol\theta_t)}` (13.14)
  "The natural state-value-function learning method to pair with this is semi-gradient TD(0)." (p.331)
- Eq. 13.18, Sec. 13.7, p.335 (PDF 357): `p(x) \doteq \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)`
  ("pi here is just the number pi ~ 3.14159"; "p(x) is the density ... It can be greater than 1").
- Eq. 13.19, p.335: `\pi(a|s,\boldsymbol\theta) \doteq \frac{1}{\sigma(s,\boldsymbol\theta)\sqrt{2\pi}}\exp\left(-\frac{(a-\mu(s,\boldsymbol\theta))^2}{2\sigma(s,\boldsymbol\theta)^2}\right)`
  with mu : S x R^{d'} -> R and sigma : S x R^{d'} -> R^+ "two parameterized function approximators".
- Eq. 13.20, p.336 (PDF 358): theta = [theta_mu, theta_sigma]^T;
  `\mu(s,\boldsymbol\theta) \doteq \boldsymbol\theta_\mu^\top\mathbf{x}_\mu(s)` and
  `\sigma(s,\boldsymbol\theta) \doteq \exp\Big(\boldsymbol\theta_\sigma^\top\mathbf{x}_\sigma(s)\Big)`.
  "The standard deviation must always be positive and is better approximated as the exponential of a linear function."
- Exercise 13.4, p.336, unnumbered eligibility vectors for the Gaussian policy:
  `\nabla\ln\pi(a|s,\boldsymbol\theta_\mu) = \frac{\nabla\pi(a|s,\boldsymbol\theta_\mu)}{\pi(a|s,\boldsymbol\theta)} = \frac{1}{\sigma(s,\boldsymbol\theta)^2}\big(a - \mu(s,\boldsymbol\theta)\big)\mathbf{x}_\mu(s)`
  `\nabla\ln\pi(a|s,\boldsymbol\theta_\sigma) = \frac{\nabla\pi(a|s,\boldsymbol\theta_\sigma)}{\pi(a|s,\boldsymbol\theta)} = \left(\frac{\big(a - \mu(s,\boldsymbol\theta)\big)^2}{\sigma(s,\boldsymbol\theta)^2} - 1\right)\mathbf{x}_\sigma(s)`

## 7. Settings
Example 13.1, "Short corridor with switched actions" (p.323, PDF 345):
| item | value (source wording) |
|---|---|
| layout | inset: S, a middle state with switched arrows, a third state, G (terminal) |
| reward | "The reward is -1 per step, as usual." |
| states, actions | "In each of the three nonterminal states there are only two actions, right and left." |
| dynamics | usual in the first and third states ("left causes no movement in the first state"); "in the second state they are reversed, so that right moves to the left and left moves to the right" |
| features | x(s, right) = [1, 0]^T and x(s, left) = [0, 1]^T, "for all s" (all states look identical) |
| epsilon-greedy | right (or left) "with high probability 1 - epsilon/2 on all steps" |
| epsilon = 0.1 values | "these two policies achieve a value (at the start state) of less than -44 and -82, respectively" (right, then left) |
| optimum | "The best probability is about 0.59, which achieves a value of about -11.6." |
| plot | J(theta) = v_{pi_theta}(S) vs "probability of right action" (0 to 1) |
Exercise 13.1 (p.324) asks for the exact optimal probability; the text does not give it.
Derived by us (not printed; label as our computation): J(p) = -2(2 - p)/(p(1 - p)) for p = P(right); p* = 2 - sqrt(2)
~ 0.586; J(p*) = -(6 + 4 sqrt(2)) ~ -11.657; J(0.95) ~ -44.21; J(0.05) ~ -82.11. Checked by solving the 3-state
linear system. These agree with the printed "about 0.59", "about -11.6", "less than -44 and -82".

Figure 13.1 (p.328, PDF 350): REINFORCE on the short corridor.
| item | value |
|---|---|
| step sizes (figure legend) | alpha = 2^-13, 2^-14, 2^-12 |
| y-axis label | "G_0 Total reward on episode averaged over 100 runs" |
| x-axis | Episode, 1 to 1000 (printed ticks) |
| reference line | v_*(s_0) (dashed) |
| caption finding | "With a good step size, the total reward per episode approaches the optimal value of the start state." |
Qualitative only (from the curves, no numbers): 2^-13 rises fastest; 2^-14 rises more slowly; 2^-12 levels off
well below the optimum.

Figure 13.2 (p.330, PDF 352): REINFORCE with vs without baseline.
| item | value |
|---|---|
| with baseline (figure legend) | alpha^theta = 2^-9, alpha^w = 2^-6 |
| plain REINFORCE (figure legend) | alpha = 2^-13 |
| baseline function (p.331) | "Here the approximate state-value function used in the baseline is v-hat(s,w) = w. That is, w is a single component, w." |
| y-axis label | "averaged over 100 runs" |
| x-axis | Episode, 1 to 1000 |
| caption finding | "Adding a baseline to REINFORCE can make it learn much faster, as illustrated here on the short-corridor gridworld (Example 13.1). The step size used here for plain REINFORCE is that at which it performs best (to the nearest power of two; see Figure 13.1)." |
Note: p.331 prints "short-corridor gridword" (typo in the book).

## 8. Findings worth remembering
- Soft-max preferences "do not approach specific values; instead they are driven to produce the optimal stochastic
  policy. If the optimal policy is deterministic, then the preferences of the optimal actions will be driven
  infinitely higher than all suboptimal actions (if permitted by the parameterization)." (p.323)
- REINFORCE convergence (p.329): "By construction, the expected update over an episode is in the same direction as
  the performance gradient. This assures an improvement in expected performance for sufficiently small alpha, and
  convergence to a local optimum under standard stochastic approximation conditions for decreasing alpha. However,
  as a Monte Carlo method REINFORCE may be of high variance and thus produce slow learning."
- REINFORCE "is well defined only for the episodic case with all updates made in retrospect after the episode is
  completed" (p.327).
- Actor-critic bias (p.331): "Note that the bias in the gradient estimate is not due to bootstrapping as such; the
  actor would be biased even if the critic was learned by a Monte Carlo method."
- Why the critic helps (p.331): REINFORCE's baseline "is made prior to the transition's action and thus cannot be
  used to assess that action"; the one-step return "is often superior to the actual return in terms of its variance
  and computational congeniality, even though it introduces bias."
- Summary (p.337): "Adding a state-value function as a baseline reduces REINFORCE's variance without introducing
  bias." "The critic introduces bias into the actor's gradient estimates, but is often desirable for the same reason
  that bootstrapping TD methods are often superior to Monte Carlo methods (substantially reduced variance)."
- n-step / lambda generalization (p.332): "The one-step return in (13.12) is merely replaced by G_{t:t+n} or
  G^lambda_t respectively."
- History (bib remarks p.337-338, PDF 359-360):
  - "REINFORCE is due to Williams (1987, 1992)." "The baseline was introduced in Williams's (1987, 1992) original
    work."
  - "The policy gradient theorem here and on page 334 was first obtained by Marbach and Tsitsiklis (1998, 2001) and
    then independently by Sutton et al. (2000)." Presentation "based primarily on that by Sutton, McAllester, Singh,
    and Mansour (2000), who introduced the term 'policy gradient methods.'"
  - "Thomas (2014) first realized that the factor of gamma^t, as specified in the boxed algorithms of this chapter,
    was needed in the case of discounted episodic problems."
  - "Example 13.1 and the results with it in this chapter were developed with Eric Graves."
  - Actor-critic: among the earliest RL methods (Witten, 1977; Barto, Sutton, and Anderson, 1983; Sutton, 1984); "The
    algorithms presented here are based on the work of Degris, White, and Sutton (2012). Actor-critic methods are
    sometimes referred to as advantage actor-critic ('A2C') methods in the literature."
  - 13.7: "The first to show how continuous actions could be handled this way appears to have been Williams (1987,
    1992)."
  - p.327 prints "(Willams, 1992)" (typo in the book).

## 9. Limitations stated
- REINFORCE: high variance, slow learning (p.329); episodic only (p.327).
- alpha^theta is hard to set (p.330).
- Summary (p.337): policy-gradient methods "are less well understood in some respects".

## 10. Outline feedback
- ch-pg equation numbers are right: Eq. 13.5 (theorem), Eq. 13.8 (REINFORCE), Eq. 13.11 (with baseline). Add
  Eq. 13.2 for the soft-max policy, and Eq. 13.9 if the worked 2-action example uses linear preferences.
- Eq. 13.5 is "proportional to", not "=". The constant is the average episode length (episodic) and 1 (continuing).
  Sutton et al. 2000 (Theorem 1, Eq. 2) write "=" because their d^pi is an unnormalized discounted state weighting.
  The page cites both; say why one has "=" and the other "proportional to".
- gamma^t: Eq. 13.8 has no gamma^t (text uses gamma = 1); all three boxes carry it (REINFORCE: alpha gamma^t G;
  baseline: alpha^theta gamma^t delta; one-step AC: I <- gamma I). With gamma < 1 the two forms give different
  updates. The page's NumPy and PyTorch REINFORCE must pick one and say which.
- Reward-to-go is already inside Eq. 13.8 (G_t is the return from t). Williams's episodic REINFORCE (his Eq. 11)
  used the whole-episode reinforcement r for every step; see williams1992.md. Frame "reward-to-go" as S&B's form,
  and Williams's sum as the cruder original.
- viz-corridor: quote the book ("about 0.59", "about -11.6", "less than -44 and -82") and, if the page computes J(p),
  label the computed values as ours (p* ~ 0.586, J* ~ -11.66). The book's "-11.6" is a truncation of -11.657, so a
  computed readout of -11.66 will not match the book's string. Show both, or show the book's.
- Figure 13.1 and 13.2 step sizes are legend text: 2^-13, 2^-14, 2^-12; with baseline alpha^theta = 2^-9,
  alpha^w = 2^-6, baseline v-hat(s,w) = w (one weight), 100 runs, 1000 episodes. If the page reproduces Figure 13.2,
  these are the settings. The finding is qualitative: the baseline learns "much faster".
- ch-ac: Eq. 13.12-13.14 and the one-step AC box are the S&B anchor. Add the p.331 quote that the actor is biased
  "even if the critic was learned by a Monte Carlo method" to the confusion block, and the bib note that S&B calls
  these "advantage actor-critic ('A2C')" methods.
- ch-continuous (Gaussian part): Eq. 13.18 (density), Eq. 13.19 (policy), Eq. 13.20 (mean linear, sigma = exp of
  linear), Exercise 13.4 (eligibility vectors, unnumbered). For "with numbers", the sigma eligibility is
  ((a - mu)^2/sigma^2 - 1) x_sigma: it is positive when |a - mu| > sigma.
