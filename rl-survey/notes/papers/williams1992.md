# Williams 1992: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning

## 1. Identity
- Ronald J. Williams (sole author; first = last author), College of Computer Science, Northeastern University.
- Venue line printed on p.1 of the file: "Appears in Machine Learning, 8, pp. 229-256, 1992." S&B reference list
  (p.517, PDF 539) adds the issue: "Machine Learning, 8(3-4):229-256".
- File read: sources/classic/williams1992-reinforce.pdf, 27 pages, a preprint layout with its own page numbers 1-27
  (journal page numbers do not appear). All page numbers below are this file's printed page numbers.
- Sections: 1 Introduction (p.1); 2 Reinforcement-Learning Connectionist Networks (p.3); 3 The Expected
  Reinforcement Performance Criterion (p.5); 4 REINFORCE Algorithms (p.5); 5 Episodic REINFORCE Algorithms (p.7);
  6 REINFORCE With Multiparameter Distributions (p.9); 7 Compatibility with Backpropagation (p.11; 7.1 Networks Using
  Deterministic Hidden Units p.11; 7.2 Backpropagating Through Random Number Generators p.12); 8 Algorithm Performance
  and Other Issues (p.13; 8.1 Convergence Properties p.13; 8.2 Gaussian Unit Search Behavior p.15; 8.3 Choice of
  Reinforcement Baseline p.16; 8.4 Alternate Forms for Eligibility p.16; 8.5 Use of Other Local Gradient Estimates
  p.17); 9 Conclusion (p.18); Appendix A, proofs of Theorems 1 and 2 (p.19-22); Appendix B, proof of Proposition 1.
- Notation: a network of stochastic units; unit i has input x^i, weights w^i (entries w_ij), output y_i drawn from
  g_i(xi, w^i, x^i) = Pr{y_i = xi | w^i, x^i} (p.3). r is the scalar reinforcement for a trial (immediate
  reinforcement: one time step per trial, p.3). No R_{t+1} index shift arises; in the episodic case, r is delivered
  at the end of a k-step episode (p.8).

## 2. Core problem
Find learning rules for networks of stochastic units that climb the gradient of expected reinforcement E{r|W}
without computing that gradient explicitly or storing what would be needed to compute it (abstract, p.1; p.2).

## 3. What prior methods did and why that hurt (per the source)
Existing stochastic learning automata and connectionist RL rules (L_R-I, A_R-I, A_R-P, reinforcement comparison)
were not derived from a gradient. p.2: "to the extent that certain existing algorithms resemble the algorithms arising
from such a gradient analysis, our understanding of them may be enhanced." The paper shows that L_R-I (p.6) and A_R-I
on Bernoulli-logistic units (p.7) are REINFORCE algorithms. Delayed reinforcement was usually handled by pairing an
immediate-reinforcement learner with a TD critic (p.2).

## 4. The one key idea
Update each weight by (reinforcement minus a baseline) times the characteristic eligibility d ln g / d w. In
expectation this is the gradient of expected reinforcement (Theorem 1).

## 5. Mechanism, step by step
1. Performance measure: E{r|W} (Sec. 3, p.5).
2. REINFORCE update (Sec. 4, p.5, unnumbered): `\Delta w_{ij} = \alpha_{ij}(r - b_{ij})e_{ij}`, with
   alpha_ij = "learning rate factor", b_ij = "reinforcement baseline", `e_{ij} = \partial\ln g_i/\partial w_{ij}` =
   "characteristic eligibility of w_ij". Conditions: "the reinforcement baseline b_ij is conditionally independent of
   y_i, given W and x^i, and the rate factor alpha_ij is nonnegative and depends at most on w^i and t. (Typically,
   alpha_ij will be taken to be a constant.)" (p.5)
3. Bernoulli-logistic unit: e_ij = (y_i - p_i) x_j (Eq. 7, p.7); with b = 0: Delta w_ij = alpha r (y_i - p_i) x_j
   (Eq. 8, p.7).
4. Reinforcement comparison baseline (p.7): `\Delta w_{ij} = \alpha(r - \bar r)(y_i - p_i)x_j` (Eq. 9), "which is
   then a REINFORCE algorithm as long as the computation of r-bar is never based on the current value of y_i (or the
   current value of r)", with `\bar r(t) = \gamma r(t-1) + (1-\gamma)\bar r(t-1)` (Eq. 10), 0 < gamma <= 1. Here
   gamma is an averaging rate, not a discount.
5. Episodic REINFORCE (Sec. 5, p.8): `\Delta w_{ij} = \alpha_{ij}(r - b_{ij})\sum_{t=1}^{k}e_{ij}(t)` (Eq. 11), with
   r delivered at the end of the episode. Theorem 2 gives the same guarantee as Theorem 1.
6. Gaussian unit (Sec. 6, p.9-10): learn mean mu and standard deviation sigma (see equations).

## 6. Equations (from the page images)
- REINFORCE, Sec. 4, p.5, unnumbered: `\Delta w_{ij} = \alpha_{ij}(r - b_{ij})e_{ij}`.
- Acronym (quote, p.5): "Any learning algorithm having this particular form will be called a REINFORCE algorithm. The
  name is an acronym for 'REward Increment = Nonnegative Factor times Offset Reinforcement times Characteristic
  Eligibility,' which describes the form of the algorithm."
- Theorem 1 (p.5): "For any REINFORCE algorithm, the inner product of E{Delta W|W} and grad_W E{r|W} is nonnegative.
  Furthermore, if alpha_ij > 0 for all i and j, then this inner product is zero only when grad_W E{r|W} = 0. Also,
  if alpha_ij = alpha is independent of i and j, then E{Delta W|W} = alpha grad_W E{r|W}."
  Plain words: on average, every REINFORCE step points uphill on expected reinforcement; with one shared step size it
  is exactly the gradient times alpha. p.6: "The last sentence of the theorem is equivalent to the claim that for each
  weight w_ij the quantity (r - b_ij) d ln g_i / d w_ij represents an unbiased estimate of dE{r|W}/dw_ij." Proof in
  Appendix A (Theorem 1 restated p.21 as E{Delta W|W}^T grad_W E{r|W} >= 0).
- Baseline adds no bias (Appendix A, p.19-20): Fact 2 `\sum_{\xi\in Y_i}\partial g_i/\partial w_{ij}(\xi,\mathbf{w}^i,\mathbf{x}^i) = 0`,
  because the probabilities sum to 1. In Lemma 1's proof the baseline term is
  E{b_ij|W, x^i} sum_xi dg_i/dw_ij = 0 "by Fact 2", using "E{b_ij|W, x^i, y_i = xi} = E{b_ij|W, x^i}, by assumption".
  Lemma 1: E{Delta w_ij|W, x^i} = alpha_ij dE{r|W, x^i}/dw_ij; Lemma 2 removes the conditioning on x^i.
- Theorem 2 (p.8): "For any episodic REINFORCE algorithm, the inner product of E{Delta W|W} and grad_W E{r|W} is
  nonnegative. Furthermore, if alpha_ij > 0 for all i and j, then this inner product is zero only when
  grad_W E{r|W} = 0. Also, if alpha_ij = alpha is independent of i and j, then E{Delta W|W} = alpha grad_W E{r|W}."
- Gaussian unit (Sec. 6): `g(y,\mu,\sigma) = \frac{1}{(2\pi)^{1/2}\sigma}e^{-\frac{(y-\mu)^2}{2\sigma^2}}` (p.9, unnumbered);
  `\frac{\partial\ln g}{\partial\mu} = \frac{y-\mu}{\sigma^2}` (Eq. 12, p.9);
  `\frac{\partial\ln g}{\partial\sigma} = \frac{(y-\mu)^2-\sigma^2}{\sigma^3}` (p.10, unnumbered);
  `\Delta\mu = \alpha_\mu(r - b_\mu)\frac{y-\mu}{\sigma^2}` (Eq. 13, p.10);
  `\Delta\sigma = \alpha_\sigma(r - b_\sigma)\frac{(y-\mu)^2-\sigma^2}{\sigma^3}` (Eq. 14, p.10);
  "A reasonable algorithm is obtained by setting alpha_mu = alpha_sigma = alpha sigma^2" with b_mu = b_sigma from
  reinforcement comparison (p.10).
- Footnote 2 (p.10): "Strictly speaking, there is no choice of alpha for this algorithm guaranteeing that sigma will
  not become negative, unless the normal distribution has its tails truncated (which is necessarily the case in
  practice). Another approach is to take lambda = ln sigma as the adaptable parameter rather than sigma, which leads to
  an algorithm guaranteed to keep sigma positive."
- Sec. 7.2 (p.13), backpropagating through a random number generator: "y = mu + sigma z, where z is a standard normal
  deviate", so dy/dmu = 1 and dy/dsigma = z = (y - mu)/sigma.

## 7. Settings
No numbered benchmark. Experiments are described qualitatively (Sec. 8): Sutton (1984) single-unit studies, Williams
and Peng (1991), Gaussian-unit function optimization.

## 8. Findings worth remembering
- p.9: "control over sigma is tantamount to control over the unit's exploratory behavior."
- Sec. 8.1 (p.13-14): "If such an algorithm does converge, one might expect it to converge to a local maximum, but
  there need be no such convergence." "Like A_R-P, REINFORCE is generally very slow even when it succeeds. Episodic
  REINFORCE has been found to be especially slow, but this, too, is not surprising since it performs temporal
  credit-assignment by essentially spreading credit or blame uniformly over all past times."
- p.14: in Sutton's (1984) studies, "REINFORCE with reinforcement comparison was found to outperform all other
  algorithms investigated." Williams and Peng (1991): "One particularly interesting variant incorporated an entropy
  term in the reinforcement signal".
- Sec. 8.2 (p.15): with r >= 0 and no reinforcement comparison (b = 0), "REINFORCE may cause sigma to converge to 0
  before mu has moved to the top of any hill."
- Sec. 8.3 (p.16), baseline: "One important limitation of the analysis given here is that it offers no basis for
  choosing among various choices of reinforcement baseline in REINFORCE algorithms. While Theorem 1 applies equally
  well to any such choice, extensive empirical investigation of such algorithms leads to the inescapable conclusion
  that use of an adaptive reinforcement baseline incorporating something like the reinforcement comparison strategy
  can greatly enhance convergence speed, and, in some cases, can lead to a big difference in qualitative behavior as
  well." A variance-minimizing baseline (Williams 1986; Dayan 1990) is "not the mean reinforcement" and gave "a slight
  improvement in convergence speed".
- p.7: "it is generally believed that the use of reinforcement comparison leads to algorithms having superior
  performance in general."
- Reward per step (p.9): for reinforcement at every step, one can "simply replace r in 11 by sum_{t=1}^k r(t)", but
  when r is causal "there is a potentially better way to perform the necessary credit assignment ... treat this
  learning problem over the k-time-step interval as k different but overlapping episodic learning problems, all
  starting at the beginning of the episode. We omit further discussion of the details of this approach."

## 9. Limitations stated
- Conclusion (p.18): "The main disadvantages are the lack of a general convergence theory applicable to this class of
  algorithms and, as with all gradient algorithms, an apparent susceptibility to convergence to false optima."
- No guidance on choosing the baseline (Sec. 8.3).
- Later work that fixed it (per sources here): Sutton et al. 2000 use a learned (compatible) value function to reduce
  variance; S&B ch. 13 adds a learned state-value baseline and actor-critic.

## 10. Outline feedback
- ch-pg "Where this came from": REINFORCE acronym quote (p.5); Theorem 1 = "the average update is the gradient"; the
  baseline appears from the start (b_ij, p.5) and S&B credit it to Williams (S&B p.338).
- "Reward-to-go": Williams's episodic rule (Eq. 11) multiplies the whole-episode r by the summed eligibilities. The
  "k overlapping problems" remark (p.9) is the seed of reward-to-go, and Williams left it undeveloped. This is a
  clean "one fix at a time" story for the chapter.
- Baseline unbiasedness: Williams's Fact 2 (sum of dg/dw over outputs = 0) is the same identity the outline plans to
  derive (E[grad log pi] = 0). The condition is that the baseline does not depend on the sampled output (p.5).
- ch-continuous: Williams Sec. 6 is the first Gaussian policy (S&B p.338 agrees: "appears to have been Williams (1987,
  1992)"). Footnote 2 already proposes learning ln sigma, which is what modern code does. Sec. 7.2 writes
  y = mu + sigma z and differentiates through it, an early form of the reparameterization used by SAC.
- ch-ac/ch-ppo (entropy bonus history): p.14 mentions an entropy term in the reinforcement signal (Williams and Peng
  1991). One line in "Where this came from" if wanted.
