# S&B Chapter 3: Finite Markov Decision Processes

## 1. Identity
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018; 2020 printing read). See
  sb-ch01.md for the full book identity. Book page = PDF page - 22.
- Chapter 3 "Finite Markov Decision Processes", book pp. 47-71 (PDF 69-93).
- Sections: 3.1 The Agent-Environment Interface p.47; 3.2 Goals and Rewards p.53; 3.3 Returns and Episodes p.54;
  3.4 Unified Notation for Episodic and Continuing Tasks p.57; 3.5 Policies and Value Functions p.58; 3.6 Optimal
  Policies and Optimal Value Functions p.62; 3.7 Optimality and Approximation p.67; 3.8 Summary p.68;
  Bibliographical and Historical Remarks p.69.

## 2. Core problem
Bandits plus an associative aspect: actions influence immediate rewards and also later states, and through them
later rewards. So the agent must trade off immediate and delayed reward. State-dependent values q*(s, a) and
v*(s) "are essential to accurately assigning credit for long-term consequences to individual action selections"
(ch. 3 opening, p.47).

## 3. What prior methods did and why that hurt (per the source)
- Bandits estimate q*(a) per action only; MDPs need values per state or state-action pair (p.47).
- MDP literature usually writes dynamics as p(s'|s, a) and expected rewards r(s, a). The book uses the joint
  p(s', r|s, a) because "In reinforcement learning, however, we more often have to refer to individual actual or
  sample rewards (rather than just their expected values)" and it makes plain that S_t and R_t are "jointly
  determined, and thus must have the same time index" (bib 3.1, p.70).

## 4. The one key idea
Write the interaction as states, actions and rewards with Markov dynamics p(s', r|s, a). Define the (discounted)
return and the value functions of a policy. Values satisfy recursive Bellman equations; optimal values satisfy
the Bellman optimality equations, and acting greedily on v* (or on q*, with no model) is optimal.

## 5. Mechanism, step by step
1. Loop (3.1, p.48): at t = 0, 1, 2, ... the agent gets S_t, picks A_t; "One time step later, in part as a
   consequence of its action, the agent receives a numerical reward, R_{t+1}, and finds itself in a new state,
   S_{t+1}." Trajectory Eq. 3.1.
2. Dynamics p (Eq. 3.2), sums to 1 (Eq. 3.3); derived p(s'|s, a) (3.4), r(s, a) (3.5), r(s, a, s') (3.6).
3. Markov property: a restriction on the state (p.49, quoted in section 8).
4. Goal: reward hypothesis (3.2, p.53).
5. Return: episodic sum (3.7); discounted (3.8); recursion (3.9) with G_T = 0; constant-reward sum (3.10);
   unified form with an absorbing state (3.11).
6. Policy pi(a|s): "a mapping from states to probabilities of selecting each possible action" (p.58). Values
   v_pi (3.12), q_pi (3.13). Bellman equation for v_pi (3.14), backup diagram (p.59).
7. Optimality: v* (3.15), q* (3.16), q* from v* (3.17), Bellman optimality for v* (3.18, 3.19), for q* (3.20).
   Greedy w.r.t. v* is optimal; q* needs no model to act (pp.64-65).

## 6. Equations (transcribed from the PDF; all verified on the rendered pages)
- Eq. 3.1 (p.48): `S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \ldots`
- Eq. 3.2 (p.48, PDF 70): `p(s', r \mid s, a) \doteq \Pr\{S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a\}`,
  for all s', s in S, r in R, a in A(s). "The function p defines the dynamics of the MDP." The dot over "=" marks
  a definition. p : S x R x S x A -> [0, 1].
- Eq. 3.3 (p.49): `\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1`, for all s in S, a in A(s).
- Eq. 3.4 (p.49): `p(s' \mid s, a) \doteq \Pr\{S_t = s' \mid S_{t-1} = s, A_{t-1} = a\} = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)`
- Eq. 3.5 (p.49): `r(s, a) \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)`
- Eq. 3.6 (p.49): `r(s, a, s') \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a, S_t = s'] = \sum_{r \in \mathcal{R}} r \frac{p(s', r \mid s, a)}{p(s' \mid s, a)}`
- Eq. 3.7 (3.3, p.54, PDF 76): `G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T`, "where T is a final time
  step". Episodic tasks; T "is a random variable that normally varies from episode to episode".
- Eq. 3.8 (p.55, PDF 77): `G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}`,
  "where gamma is a parameter, 0 <= gamma <= 1, called the discount rate." The agent "chooses A_t to maximize the
  expected discounted return".
- Eq. 3.9 (p.55):
  `G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots = R_{t+1} + \gamma\big(R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots\big) = R_{t+1} + \gamma G_{t+1}`
  "Returns at successive time steps are related to each other in a way that is important for the theory and
  algorithms of reinforcement learning". "Note that this works for all time steps t < T, even if termination
  occurs at t + 1, provided we define G_T = 0. This often makes it easy to compute returns from reward sequences."
- Eq. 3.10 (p.55): `G_t = \sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma}`, for a constant reward of +1 and
  gamma < 1.
- Eq. 3.11 (3.4, p.57, PDF 79): `G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k`, "including the possibility that
  T = infinity or gamma = 1 (but not both)". Episode end = "a special absorbing state that transitions only to
  itself and that generates only rewards of zero".
- Eq. 3.12 (3.5, p.58, PDF 80):
  `v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\Big[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\Big|\, S_t = s\Big]`, for all s in S.
  "the expected return when starting in s and following pi thereafter"; "the value of the terminal state, if any,
  is always zero"; "the state-value function for policy pi".
- Eq. 3.13 (p.58):
  `q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\Big[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\Big|\, S_t = s, A_t = a\Big]`
  "the action-value function for policy pi".
- Eq. 3.14 (p.59, PDF 81), Bellman equation for v_pi:
  ```
  v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s]
           = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]                                   (by (3.9))
           = \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r|s, a)\Big[r + \gamma\mathbb{E}_\pi[G_{t+1}|S_{t+1} = s']\Big]
           = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)\Big[r + \gamma v_\pi(s')\Big],  for all s \in S   (3.14)
  ```
  "It expresses a relationship between the value of a state and the values of its successor states." "It states
  that the value of the start state must equal the (discounted) value of the expected next state, plus the reward
  expected along the way." "The value function v_pi is the unique solution to its Bellman equation." (pp.59-60)
  Gloss: pi(a|s) = chance of action a in s; p(s', r|s, a) = chance of next state s' and reward r; gamma =
  discount; s' ranges over S+ in episodic tasks.
- Eq. 3.15 (3.6, p.62): `v_*(s) \doteq \max_\pi v_\pi(s)`, for all s in S.
- Eq. 3.16 (p.63, PDF 85): `q_*(s, a) \doteq \max_\pi q_\pi(s, a)`, for all s in S and a in A(s).
- Eq. 3.17 (p.63): `q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]`.
- Eqs. 3.18-3.19 (p.63), Bellman optimality equation for v*:
  ```
  v_*(s) = \max_{a \in A(s)} q_{\pi_*}(s, a)
         = \max_a \mathbb{E}_{\pi_*}[G_t \mid S_t = s, A_t = a]
         = \max_a \mathbb{E}_{\pi_*}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a]     (by (3.9))
         = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]         (3.18)
         = \max_a \sum_{s', r} p(s', r|s, a)\big[r + \gamma v_*(s')\big]                  (3.19)
  ```
  "the value of a state under an optimal policy must equal the expected return for the best action from that
  state". "The last two equations are two forms of the Bellman optimality equation for v*." (p.64)
- Eq. 3.20 (p.64, PDF 86), Bellman optimality equation for q*:
  `q_*(s, a) = \mathbb{E}\Big[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \,\Big|\, S_t = s, A_t = a\Big] = \sum_{s', r} p(s', r|s, a)\Big[r + \gamma \max_{a'} q_*(s', a')\Big]`
  Backup diagrams (Figure 3.4): same as for v_pi, q_pi "except that arcs have been added at the agent's choice
  points to represent that the maximum over that choice is taken rather than the expected value given some
  policy." "For finite MDPs, the Bellman optimality equation for v* (3.19) has a unique solution." "if there are n
  states, then there are n equations in n unknowns" (nonlinear, because of the max) (p.64).

## 7. Config / example settings
Example 3.5 Gridworld (3.5, p.60, PDF 82; Figure 3.2):

| Setting | Value |
|---|---|
| Grid | 5 x 5 cells (Figure 3.2; text says "a rectangular gridworld") |
| Actions | north, south, east, west; move one cell, deterministic |
| Off-grid move | location unchanged, reward -1 |
| Other moves | reward 0, except from A and B |
| From A | all four actions: reward +10, move to A' |
| From B | all actions: reward +5, move to B' |
| Positions (row, col from top-left, 0-based; read from Figure 3.2) | A (0,1), A' (4,1), B (0,3), B' (2,3) |
| Policy | equiprobable random ("selects all four actions with equal probability in all states") |
| gamma | 0.9 |
| Solved by | "solving the system of linear equations (3.14)" |

Figure 3.2 (right), v_pi for the random policy, as printed (one decimal):
```
 3.3  8.8  4.4  5.3  1.5
 1.5  3.0  2.3  1.9  0.5
 0.1  0.7  0.7  0.4 -0.4
-1.0 -0.4 -0.4 -0.6 -1.2
-1.9 -1.3 -1.2 -1.4 -2.0
```
Exercise 3.14 (p.60): the center state is valued +0.7; its four neighbors are +2.3, +0.4, -0.4, and +0.7. "(These
numbers are accurate only to one decimal place.)"

Example 3.8 (p.65, PDF 87; Figure 3.5 middle), v* for the same gridworld, as printed:
```
22.0 24.4 22.0 19.4 17.5
19.8 22.0 19.8 17.8 16.0
17.8 19.8 17.8 16.0 14.4
16.0 17.8 16.0 14.4 13.0
14.4 16.0 14.4 13.0 11.7
```
Example 3.8 does not restate gamma ("the simple grid task introduced in Example 3.5"). Exercise 3.24 (p.67):
"Figure 3.5 gives the optimal value of the best state of the gridworld as 24.4, to one decimal place." Our
arithmetic (not printed): with gamma = 0.9 the loop A -> A' -> 4 moves up -> A takes 5 steps, so
v*(A) = 10/(1 - 0.9^5) = 24.419, consistent with 24.4. Figure 3.5 (right) shows the optimal policies as arrows.

Other examples with settings:
- Example 3.4 Pole-Balancing (p.56): episodic version, reward "+1 for every time step on which failure did not
  occur, so that the return at each time would be the number of steps until failure"; continuing version, reward
  "-1 on each failure and zero at all other times", return "related to -gamma^{K-1}, where K is the number of time
  steps before failure".
- Reward examples (3.2, p.53): maze escape, "the reward is often -1 for every time step that passes prior to
  escape"; chess and checkers, "+1 for winning, -1 for losing, and 0 for drawing and for all nonterminal
  positions".
- Example 3.3 Recycling Robot (p.52) and Example 3.6 Golf (p.61): not needed by the outline.

## 8. Results and findings worth remembering
- Markov property (3.1, p.49, PDF 71): "In a Markov decision process, the probabilities given by p completely
  characterize the environment's dynamics. That is, the probability of each possible value for S_t and R_t depends
  on the immediately preceding state and action, S_{t-1} and A_{t-1}, and, given them, not at all on earlier states
  and actions. This is best viewed as a restriction not on the decision process, but on the state. The state must
  include information about all aspects of the past agent-environment interaction that make a difference for the
  future. If it does, then the state is said to have the Markov property."
- Index convention, footnote 4 (p.48, PDF 70): "We use R_{t+1} instead of R_t to denote the reward due to A_t
  because it emphasizes that the next reward and next state, R_{t+1} and S_{t+1}, are jointly determined.
  Unfortunately, both conventions are widely used in the literature."
- Reward hypothesis (3.2, p.53, PDF 75): "That all of what we mean by goals and purposes can be well thought of as
  the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)."
  "The use of a reward signal to formalize the idea of a goal is one of the most distinctive features of
  reinforcement learning."
- Reward design (p.54): "the reward signal is not the place to impart to the agent prior knowledge about how to
  achieve what we want it to do." Chess: reward winning, not taking pieces, else "it might find a way to take the
  opponent's pieces even at the cost of losing the game. The reward signal is your way of communicating to the
  agent what you want achieved, not how you want it achieved."
- Discount (p.55): "a reward received k time steps in the future is worth only gamma^{k-1} times what it would be
  worth if it were received immediately." gamma = 0: "myopic". "As gamma approaches 1, the return objective takes
  future rewards into account more strongly; the agent becomes more farsighted."
- Episodic vs continuing (3.3, p.54): episodes end in "a special state called the terminal state"; continuing
  tasks "go on continually without limit", where (3.7) fails because "the final time step would be T = infinity,
  and the return ... could easily be infinite."
- Example 3.5 reading (p.60): negative values near the lower edge come from "the high probability of hitting the
  edge of the grid there under the random policy"; "A's expected return is less than its immediate reward of 10,
  because from A the agent is taken to state A' from which it is likely to run into the edge of the grid. State B,
  on the other hand, is valued more than its immediate reward of 5".
- Greedy on v* is optimal (p.64): "any policy that is greedy with respect to the optimal evaluation function v* is
  an optimal policy"; "a one-step-ahead search yields the long-term optimal actions" (p.65).
- q* needs no model (p.65): "the optimal action-value function allows optimal actions to be selected without
  having to know anything about possible successor states and their values, that is, without having to know
  anything about the environment's dynamics."
- Solving the BOE directly "is rarely directly useful"; it needs "(1) the dynamics of the environment are
  accurately known; (2) computational resources are sufficient to complete the calculation; and (3) the states
  have the Markov property." Backgammon has "about 10^20 states", so solving for v* "would take thousands of years
  on today's fastest computers" (p.66).
- Approximation (3.7, pp.67-68): "tabular case" defined; online RL can "put more effort into learning to make
  good decisions for frequently encountered states".
- History (bib, pp.69-71): "The most influential integration of reinforcement learning and MDPs is due to Watkins
  (1989)" (p.70); reward hypothesis suggested by Michael Littman (p.70); "The Bellman optimality equation (for v*)
  was popularized by Richard Bellman (1957a), who called it the 'basic functional equation'", continuous
  counterpart "Hamilton-Jacobi-Bellman equation" (p.71); Watkins's Q-learning made action values central, "these
  functions are often called 'Q-functions'" (p.71); Denardo (1967) treated DP "in terms of contraction mappings"
  (p.71).

## 9. Limitations the source states
- Finite MDPs and the Markov property are assumed; Part II approximations and ch. 17 relax them (p.49).
- Solving the BOE exactly needs the three assumptions above, which "are rarely true in practice" (p.66).
- Discounted returns suit continuing tasks; "Later, in Chapter 10, we will introduce a formulation that is both
  continuing and undiscounted" (p.58).

## 10. Outline feedback (ch-mdp and ch-bellman)
ch-mdp:
1. Equation numbers in the outline are right: G_t discounted = Eq. 3.8, recursion = Eq. 3.9 (state G_T = 0 with
   it). The undiscounted return used in ch-score is Eq. 3.7.
2. 1/(1 - gamma) is Eq. 3.10: the return for a constant reward of +1. The book never calls it a horizon. Keep the
   "rule of thumb" label and cite Eq. 3.10 only for the sum.
3. The book says a reward k steps ahead is worth gamma^{k-1} (k counted from 1). viz-discount draws gamma^k;
   label the axis so the index is clear.
4. Quote the Markov property (p.49) and footnote 4 (p.48) for the index convention; footnote 4 is the source
   sentence for the notation card ("both conventions are widely used").
5. Confusion "the reward is the goal": the reward hypothesis (p.53) and "what you want achieved, not how you want
   it achieved" (p.54), with the chess subgoal example.
6. Two-chargers crossover gamma^4 = 0.1 assumes zero reward on the way. If the variant keeps the yard's -0.04 step
   reward, the crossover moves (our arithmetic at gamma = 0.562, -0.04 per non-terminal step: near 0.52 vs far
   0.48). Compute it on the page, or set the step reward to 0 in that variant.
7. "Episodes vs continuing tasks": cite the absorbing-state convention and Eq. 3.11 (p.57). Example 3.4 (p.56)
   gives the CartPole-style reward (+1 per step, return = steps until failure), useful as a plant for ch. 9.
ch-bellman:
8. Equation numbers are right: Bellman for v_pi = Eq. 3.14 (p.59); optimality for v* = Eq. 3.19 (3.18 is the
   expectation form); for q* = Eq. 3.20 (p.64). Also available: v_pi def Eq. 3.12, q_pi def Eq. 3.13, v* Eq. 3.15,
   q* Eq. 3.16, q* from v* Eq. 3.17.
9. "Q_pi(s, a), and V from Q": S&B has no numbered equation for v_pi in terms of q_pi (Exercise 3.12) or q_pi in
   terms of v_pi (Exercise 3.13). q_pi from v_pi is numbered later as Eq. 4.6 (p.78). Cite Eq. 4.6, or show both
   unnumbered.
10. Notation: S&B uses lowercase v_pi, q_pi, v*, q* for true values and capital V, Q for tabular estimates
    (Preface pp. xiii-xiv; Summary of Notation p. xx). The outline's V_pi, Q_pi, Q* differ; the notation card
    should state the page's choice.
11. Self-check "compute V(s) from neighbor values": Exercise 3.14 is a ready S&B instance (center 0.7; neighbors
    2.3, 0.4, -0.4, 0.7; gamma 0.9; random policy). The book warns the numbers are accurate to one decimal, so the
    check is approximate (our arithmetic: 0.9 x (2.3 + 0.4 - 0.4 + 0.7)/4 = 0.675).
12. "For a fixed policy it is a linear system": Example 3.5 values were "computed by solving the system of linear
    equations (3.14)" (p.60); 4.1 states "(4.4) is a system of |S| simultaneous linear equations in |S| unknowns"
    (p.74).
13. "Acting from Q* needs no model": quote p.65. "Where this came from": Bellman (1957a) "basic functional
    equation" (p.71).
