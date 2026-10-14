# Sutton, McAllester, Singh, Mansour 2000: Policy Gradient Methods for Reinforcement Learning with Function Approximation

## 1. Identity
- Authors: Richard S. Sutton (first), David McAllester, Satinder Singh, Yishay Mansour (last); AT&T Labs - Research,
  Florham Park, NJ.
- File: sources/classic/sutton2000-pgtheorem.pdf, 7 pages, printed page numbers 1057-1063. The file prints no venue
  line. PDF metadata Subject: "Neural Information Processing Systems http://nips.cc/".
- Venue (S&B reference list, p.512, PDF 534): "In Advances in Neural Information Processing Systems 12, pp. 1057-1063.
  MIT Press, Cambridge, MA." Year 2000. The conference year (1999) is not printed in any file we have.
- Sections: abstract and introduction (p.1057-1058); 1 Policy Gradient Theorem (p.1058-1059); 2 Policy Gradient with
  Approximation (p.1059-1060); 3 Application to Deriving Algorithms and Advantages (p.1060-1061); 4 Convergence of
  Policy Iteration with Function Approximation (p.1061-1062); References (p.1062); Appendix: Proof of Theorem 1
  (p.1062-1063).
- Notation: pi(s, a, theta) = Pr{a_t = a | s_t = s, theta} (state first); R^a_s = E{r_{t+1} | s_t = s, a_t = a}, so
  the reward after a_t is r_{t+1}, the same convention as S&B (p.1058). Performance is rho (not J); state weighting
  d^pi(s).

## 2. Core problem
Value-function methods with function approximation lacked convergence guarantees (p.1057). Can the gradient of
performance w.r.t. policy parameters be estimated from experience with a learned value function, and does following
it converge?

## 3. What prior methods did and why that hurt (per the source)
- p.1057: the value-function approach "is oriented toward finding deterministic policies, whereas the optimal policy
  is often stochastic"; "an arbitrarily small change in the estimated value of an action can cause it to be, or not
  be, selected"; "Q-learning, Sarsa, and dynamic programming methods have all been shown unable to converge to any
  policy for simple MDPs and simple function approximators (Gordon, 1995, 1996; Baird, 1995; Tsitsiklis and van Roy,
  1996; Bertsekas and Tsitsiklis, 1996)."
- p.1058: "REINFORCE learns much more slowly than RL methods using value functions and has received relatively little
  attention. Learning a value function and using it to reduce the variance of the gradient estimate appears to be
  essential for rapid learning."

## 4. The one key idea
The performance gradient has no term in the derivative of the state distribution (Theorem 1). If the learned Q
approximator is "compatible" with the policy parameterization, using it in place of Q^pi gives the exact gradient
(Theorem 2).

## 5. Mechanism, step by step
1. Update theta approximately along the gradient: Eq. 1.
2. Theorem 1 (Eq. 2): the gradient is a d^pi-weighted sum of grad pi times Q^pi, for both average-reward and
   start-state formulations.
3. Sampling it with actual returns R_t gives Williams's REINFORCE (p.1059).
4. Fit f_w to Q^pi by least squares (Eq. 3). If grad_w f_w = grad_theta pi / pi (compatibility, Eq. 4), the error in
   f_w is orthogonal to grad pi (Eq. 6), so f_w can replace Q^pi exactly (Eq. 5, Theorem 2).
5. For a Gibbs (softmax) policy, the compatible f_w is linear in the policy features centered per state, so it is
   really an advantage function (p.1061).
6. Any state-only function v(s) can be added (a baseline): it changes variance, not the expected gradient (p.1061).
7. Theorem 3: policy iteration with compatible function approximation converges to a locally optimal policy.

## 6. Equations (from the page images; ASCII transcription)
- Eq. 1, p.1058: `\Delta\theta \approx \alpha\frac{\partial\rho}{\partial\theta}` ("alpha is a positive-definite step
  size").
- Average-reward formulation (p.1059, unnumbered):
  `\rho(\pi) = \lim_{n\to\infty}\frac{1}{n}E\{r_1 + r_2 + \cdots + r_n \mid \pi\} = \sum_s d^\pi(s)\sum_a\pi(s,a)\mathcal{R}_s^a`,
  `Q^\pi(s,a) = \sum_{t=1}^{\infty}E\{r_t - \rho(\pi) \mid s_0 = s, a_0 = a, \pi\}`; d^pi = stationary distribution.
- Start-state formulation (p.1059, unnumbered):
  `\rho(\pi) = E\Big\{\sum_{t=1}^{\infty}\gamma^{t-1}r_t \,\Big|\, s_0, \pi\Big\}`,
  `Q^\pi(s,a) = E\Big\{\sum_{k=1}^{\infty}\gamma^{k-1}r_{t+k} \,\Big|\, s_t = s, a_t = a, \pi\Big\}`,
  `d^\pi(s) = \sum_{t=0}^{\infty}\gamma^t Pr\{s_t = s \mid s_0, \pi\}` ("a discounted weighting of states"; gamma in
  [0, 1], "gamma = 1 is allowed only in episodic tasks").
- Theorem 1 (Policy Gradient), Eq. 2, p.1059: "For any MDP, in either the average-reward or start-state formulations,"
  `\frac{\partial\rho}{\partial\theta} = \sum_s d^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)`.
  It is an equality: d^pi here is unnormalized in the start-state case (S&B Eq. 13.5 uses the normalized mu and writes
  "proportional to"). Source on its meaning (p.1059): "the key aspect of both expressions for the gradient is that
  their are no terms of the form d d^pi(s)/d theta: the effect of policy changes on the distribution of states does
  not appear." (sic "their")
- REINFORCE in this notation (p.1059): `\Delta\theta_t \propto \frac{\partial\pi(s_t,a_t)}{\partial\theta}R_t\frac{1}{\pi(s_t,a_t)}`
  with R_t the actual return from t; "(the 1/pi(s_t,a_t) corrects for the oversampling of actions preferred by pi),
  which is known to follow d rho / d theta in expected value (Williams, 1988, 1992)."
- Eq. 3, p.1060 (f_w at a local optimum of squared error):
  `\sum_s d^\pi(s)\sum_a\pi(s,a)\big[Q^\pi(s,a) - f_w(s,a)\big]\frac{\partial f_w(s,a)}{\partial w} = 0`
- Theorem 2 (Policy Gradient with Function Approximation), p.1060: "If f_w satisfies (3) and is compatible with the
  policy parameterization in the sense that"
  `\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial\pi(s,a)}{\partial\theta}\frac{1}{\pi(s,a)}` (Eq. 4), "then"
  `\frac{\partial\rho}{\partial\theta} = \sum_s d^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}f_w(s,a)` (Eq. 5).
  Proof via Eq. 6: `\sum_s d^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}\big[Q^\pi(s,a) - f_w(s,a)\big] = 0`,
  "which tells us that the error in f_w(s,a) is orthogonal to the gradient of the policy parameterization."
  Footnote 1 (p.1060): "Tsitsiklis (personal communication) points out that f_w being linear in the features given on
  the righthand side may be the only way to satisfy this condition."
- Gibbs policy example (Sec. 3, p.1060-1061): `\pi(s,a) = \frac{e^{\theta^T\phi_{sa}}}{\sum_b e^{\theta^T\phi_{sb}}}`;
  compatibility requires `\frac{\partial f_w(s,a)}{\partial w} = \phi_{sa} - \sum_b\pi(s,b)\phi_{sb}`, so
  `f_w(s,a) = w^T\Big[\phi_{sa} - \sum_b\pi(s,b)\phi_{sb}\Big]`. "In this sense it is better to think of f_w as an
  approximation of the advantage function, A^pi(s,a) = Q^pi(s,a) - V^pi(s) (much as in Baird, 1993), rather than of
  Q^pi." (p.1061)
- Baseline generalization (p.1061): `\frac{\partial\rho}{\partial\theta} = \sum_s d^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}\big[f_w(s,a) + v(s)\big]`,
  v : S -> R arbitrary. "(This follows immediately because sum_a d pi(s,a)/d theta = 0, for all s in S.) The choice of
  v does not affect any of our theorems, but can substantially affect the variance of the gradient estimators. The
  issues here are entirely analogous to those in the use of reinforcement baselines in earlier work (e.g., Williams,
  1992; Dayan, 1991; Sutton, 1984). In practice, v should presumably be set to the best available approximation of
  V^pi."
- Theorem 3 (Policy Iteration with Function Approximation), p.1061: "Let pi and f_w be any differentiable function
  approximators for the policy and value function respectively that satisfy the compatibility condition (4) and for
  which max_{theta,s,a,i,j} |d^2 pi(s,a)/d theta_i d theta_j| < B < infinity. Let {alpha_k} be any step-size sequence
  such that lim_{k->inf} alpha_k = 0 and sum_k alpha_k = infinity. Then, for any MDP with bounded rewards, the sequence
  {rho(pi_k)}, defined by any theta_0, pi_k = pi(., ., theta_k), and
  w_k = w such that sum_s d^{pi_k}(s) sum_a pi_k(s,a)[Q^{pi_k}(s,a) - f_w(s,a)] d f_w(s,a)/dw = 0
  theta_{k+1} = theta_k + alpha_k sum_s d^{pi_k}(s) sum_a d pi_k(s,a)/d theta f_{w_k}(s,a),
  converges such that lim_{k->inf} d rho(pi_k)/d theta = 0."
  One sentence: alternating an exact compatible fit of f_w with a gradient step on theta drives the policy gradient
  to zero, so the policy converges to a local optimum. Proof cites Proposition 3.5, p.96 of Bertsekas and Tsitsiklis
  (1996) (p.1062).

## 7. Settings
No experiments. Purely theoretical.

## 8. Findings worth remembering
- Abstract: "Our main new result is to show that the gradient can be written in a form suitable for estimation from
  experience aided by an approximate action-value or advantage function. Using this result, we prove for the first
  time that a version of policy iteration with arbitrary differentiable function approximation is convergent to a
  locally optimal policy."
- p.1058: "Unlike the value-function approach, here small changes in theta can cause only small changes in the policy
  and in the state-visitation distribution."
- History (p.1059): "This way of expressing the gradient was first discussed for the average-reward formulation by
  Marbach and Tsitsiklis (1998), based on a related expression in terms of the state-value function due to Jaakkola,
  Singh, and Jordan (1995) and Cao and Chen (1997). We extend their results to the start-state formulation and provide
  simpler and more direct proofs. Williams's (1988, 1992) theory of REINFORCE algorithms can also be viewed as
  implying (2)." p.1058: "Konda and Tsitsiklis (in prep.) independently developed a very simialr result to ours."
  (sic) S&B (p.338) credits the term "policy gradient methods" to this paper.
- p.1058: VAPS (Baird and Moore 1999) "does not converge to a locally optimal policy, except in the case that no weight
  is put upon value-function accuracy, in which case VAPS degenerates to REINFORCE."

## 9. Limitations stated
- Theorem 3 assumes w_k solves Eq. 3 exactly at each iteration and exact expectations; it is "the first step" toward
  proving convergence of actor-critic and policy-iteration architectures (p.1058).
- Footnote 1: compatibility may force f_w to be linear in the policy's features (p.1060).

## 10. Outline feedback
- ch-pg "Where this came from": cite Theorem 1 (Eq. 2) as the policy gradient theorem; S&B Eq. 13.5 is its episodic
  textbook form. Point out "=" here vs "proportional to" in S&B (normalized vs unnormalized state weighting).
- The baseline argument (sum_a d pi / d theta = 0) appears here too (p.1061), with explicit credit to Williams's
  reinforcement baselines. Good support for the chapter's derivation.
- ch-ac: Sec. 3 is an early, citable statement that the critic should be thought of as an advantage function
  A^pi = Q^pi - V^pi (p.1061). It supports the ch-ac framing of the advantage.
- Venue: cite "NIPS 12" with year 2000, as S&B's reference list does. Avoid "NeurIPS 2000", which reads as a conference year.
  If the page says "1999", mark it as the conference year; our files do not print it.
