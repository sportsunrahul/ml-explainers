# Watkins and Dayan 1992: Q-learning (Technical Note)

Source: `sources/classic/watkins-dayan1992-qlearning.pdf` (scan with OCR, 14 PDF pages; journal page = PDF page +
278, so journal p.281 is PDF page 3). Page numbers below are journal pages. The OCR layer lost every displayed
equation; equations and the theorem were transcribed from the page images.

## 1. Identity
- Christopher J.C.H. Watkins (London) and Peter Dayan (Centre for Cognitive Science, University of Edinburgh).
- "Technical Note: Q-Learning", Machine Learning, 8, 279-292 (1992), Kluwer Academic Publishers, Boston (header,
  p.279). S&B reference list: "Machine Learning, 8(3-4):279-292" (p.516, PDF 538).
- Sections: 1 Introduction p.279; 2 The task for Q-learning p.280; 3 The convergence proof p.282 (3.1 Lemmas p.283;
  3.2 The theorem p.284); 4 Discussions and conclusions p.285; Appendix (action-replay process) p.287.

## 2. Core problem
Give a detailed proof that Q-learning, introduced in Watkins (1989), converges. Abstract (p.279): "This paper
presents and proves in detail a convergence theorem for Q-learning based on that outlined in Watkins (1989)."

## 3. What prior methods did and why that hurt
Methods that learn R_x(a) and P_xy[a] while doing DP: "any assumption of certainty equivalence, i.e., calculating
actions as if the current model were accurate, costs dearly in the early stages of learning (Barto & Singh, 1990)"
(p.280). Conclusion (p.286): such a convergence "guarantee has previously eluded most methods of reinforcement
learning."

## 4. The one key idea
Learn action values directly by a one-step bootstrapped update with a max over next actions. Abstract (p.279):
"It amounts to an incremental method for dynamic programming which imposes limited computational demands. It works
by successively improving its evaluations of the quality of particular actions at particular states." Proof device:
an artificial "action-replay process" (ARP) built from the sequence of experiences, whose optimal Q values are
exactly the learner's Q_n (Lemma A, p.283), and which converges to the real process (Lemma B).

## 5. Mechanism (p.281)
In the n-th "episode" (their word for ONE transition, not a full trial), the agent:
1. observes its current state x_n,
2. selects and performs an action a_n,
3. observes the subsequent state y_n,
4. receives an immediate payoff r_n, and
5. adjusts its Q_{n-1} values using a learning factor alpha_n by Eq. (1).
"the episodes need not form a continuous sequence - that is the y of one episode need not be the x of the next
episode" (p.281). Initial values Q_0(x, a) "are assumed given".

## 6. Equations (printed numbers)
- Value of a policy (unnumbered, p.280): `V^\pi(x) \equiv \Re_x(\pi(x)) + \gamma \sum_y P_{xy}[\pi(x)] V^\pi(y)`;
  rewards s steps ahead are discounted by gamma^s with 0 < gamma < 1. R_x(a) = mean reward, P_xy[a] = transition
  probability.
- Q values (unnumbered, p.280): `Q^\pi(x, a) = \Re_x(a) + \gamma \sum_y P_{xy}[\pi(x)] V^\pi(y)` (as printed: the
  transition term is written P_xy[pi(x)], though the text defines Q as the value of "executing action a at state
  x"; the intended term is P_xy[a]).
- Eq. (1), Q-learning update (p.281):
  ```tex
  Q_n(x, a) = \begin{cases} (1 - \alpha_n) Q_{n-1}(x, a) + \alpha_n [r_n + \gamma V_{n-1}(y_n)] & \text{if } x = x_n \text{ and } a = a_n, \\ Q_{n-1}(x, a) & \text{otherwise,} \end{cases}
  ```
- Eq. (2) (p.281): `V_{n-1}(y) \equiv \max_b \{Q_{n-1}(y, b)\}`, "the best the agent thinks it can do from state
  y."
- Notation vs S&B: the reward r_n carries the same index as x_n and a_n (S&B would write R_{t+1} after A_t).
  (1 - alpha) Q + alpha [target] is algebraically S&B's Q + alpha [target - Q] (Eq. 6.8).

Theorem (p.282), verbatim: "Given bounded rewards |r_n| <= R, learning rates 0 <= alpha_n < 1, and
```tex
\sum_{i=1}^{\infty} \alpha_{n^i(x,a)} = \infty, \qquad \sum_{i=1}^{\infty} [\alpha_{n^i(x,a)}]^2 < \infty, \qquad \forall x, a, \qquad (3)
```
then Q_n(x, a) -> Q*(x, a) as n -> infinity, for all x, a, with probability 1."
n^i(x, a) = "the index of the i-th time that action a is tried in state x" (p.281).

Conditions stated around the theorem (p.281):
- Lookup table: "this description assumes a look-up table representation for the Q_n(x, a). Watkins (1989) shows
  that Q-learning may not converge correctly for other representations."
- Exploration: "The most important condition implicit in the convergence theorem given below is that the sequence
  of episodes that forms the basis of learning must include an infinite number of episodes for each starting state
  and action." "under the stochastic conditions of the theorem, no method could be guaranteed to find an optimal
  policy under weaker conditions."
- Discount 0 < gamma < 1 in the main proof; section 4 extends it (below).

## 7. Setup
Theory only; no experiments. Finite states and actions ("discrete, finite world", p.280), controlled Markov
process, bounded rewards.

## 8. Results and findings
- Abstract (p.279): "We show that Q-learning converges to the optimum action-values with probability 1 so long as
  all actions are repeatedly sampled in all states and the action-values are represented discretely."
- Extensions sketched in section 4 (p.285-286): (a) "the non-discounted case (gamma = 1), but for a Markov process
  with absorbing goal states", and (b) "the case where many of the Q values are updated in each iteration rather
  than just one". "The convergence result holds for both of these".
- Replay remark (p.286): "If the agent can remember the details of its learning episodes, then, after altering the
  learning rates, it can use each of them more than once ... This biases the Q-learning process towards the
  particular sample of the rewards and transitions that it has experienced. In the limit of re-presenting 'old'
  cards infinitely often, this reuse amounts to the certainty equivalence step of calculating the optimal actions
  for the observed sample of the Markovian environment rather than the actual environment itself."

## 9. Limitations the source states
- p.286: "The theorem above only proves the convergence of a restricted version of Watkins' (1989) comprehensive
  Q-learning algorithm, since it does not permit updates based on the rewards from more than one iteration" (no
  TD(lambda)-style multi-step updates); "the theorem does not extend trivially to this case".
- Lookup tables only (p.281). Later work in S&B (ch. 11, deadly triad) and DQN addresses function approximation.
- History per S&B (ch. 6 remarks, p.140): Watkins (1989) introduced Q-learning and outlined the proof; "made
  rigorous by Watkins and Dayan (1992)"; more general results by Jaakkola, Jordan, and Singh (1994) and Tsitsiklis
  (1994).

## 10. Outline feedback (ch-qlearn)
- "Where this came from": Watkins 1989 is a PhD thesis ("Learning from Delayed Rewards", University of Cambridge,
  per S&B's reference list); the convergence paper is Watkins and Dayan 1992, Machine Learning 8:279-292.
- Teaching note: the paper's "episode" means a single transition (x_n, a_n, y_n, r_n). Avoid quoting it next to
  S&B's "episode" without saying so.
- Index trap for the notation card: W&D's r_n (same index as a_n) matches the deep-RL r_t convention, not S&B's
  R_{t+1}.
- The replay remark (p.286) is a sourced bridge to chapter 9's replay buffer and chapter 14's Dyna.
