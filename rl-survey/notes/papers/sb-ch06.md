# S&B 2nd ed., Chapter 6: Temporal-Difference Learning

Source: `sources/sutton-barto/RLbook2020.pdf` (2020 printing). Book page = PDF page - 22. All equations, boxes and
example numbers below were checked on rendered PDF pages (the text dump drops minus signs and gamma).

## 1. Identity
- Sutton and Barto, 2nd ed. (2018), Chapter 6 "Temporal-Difference Learning", book pp. 119-140 (PDF 141-162).
- Sections: 6.1 TD Prediction p.119; 6.2 Advantages of TD Prediction Methods p.124; 6.3 Optimality of TD(0)
  p.126; 6.4 Sarsa: On-policy TD Control p.129; 6.5 Q-learning: Off-policy TD Control p.131; 6.6 Expected Sarsa
  p.133; 6.7 Maximization Bias and Double Learning p.134; 6.8 Games, Afterstates, and Other Special Cases p.136;
  6.9 Summary p.138; Bibliographical and Historical Remarks p.139.
- Figure numbering in this printing (matters for page captions): Figure 6.1 driving home (p.123); Figure 6.2 =
  BATCH training on the random walk (Example 6.3, p.127); Figure 6.3 = Expected Sarsa / Sarsa / Q-learning vs alpha
  on the cliff (p.133); Figure 6.4 = backup diagrams for Q-learning and Expected Sarsa (p.134); Figure 6.5 =
  maximization bias MDP (p.135). The Example 6.2 graphs (p.125) and the Example 6.6 cliff graph (p.132) are
  UNNUMBERED, inside the examples.

## 2. Core problem
Learn v_pi (prediction) and then good policies (control) from raw experience, without a model, and without waiting
for the end of the episode.

## 3. What prior methods did and why that hurt
- MC must wait for the episode end before any update ("only then is G_t known", p.120); "Some applications have
  very long episodes ... Other applications are continuing tasks and have no episodes at all" (p.124).
- DP needs a model "of its reward and next-state probability distributions" (p.124).

## 4. The one key idea
Bootstrap: update V(S_t) toward R_{t+1} + gamma V(S_{t+1}) one step later. "TD methods combine the sampling of
Monte Carlo with the bootstrapping of DP" (p.120-121). "They learn a guess from a guess" (p.124).

## 5. Mechanism (boxes transcribed from PDF images)

Tabular TD(0) for estimating v_pi (box, p.120, PDF 142):
```
Input: the policy pi to be evaluated
Algorithm parameter: step size alpha in (0, 1]
Initialize V(s), for all s in S+, arbitrarily except that V(terminal) = 0
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        A <- action given by pi for S
        Take action A, observe R, S'
        V(S) <- V(S) + alpha [R + gamma V(S') - V(S)]
        S <- S'
    until S is terminal
```

Sarsa (on-policy TD control) for estimating Q ~ q_* (box, p.130, PDF 152):
```
Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
Initialize Q(s,a), for all s in S+, a in A(s), arbitrarily except that Q(terminal, .) = 0
Loop for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
        Q(S,A) <- Q(S,A) + alpha [R + gamma Q(S',A') - Q(S,A)]
        S <- S'; A <- A';
    until S is terminal
```

Q-learning (off-policy TD control) for estimating pi ~ pi_* (box, p.131, PDF 153):
```
Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
Initialize Q(s,a), for all s in S+, a in A(s), arbitrarily except that Q(terminal, .) = 0
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        Take action A, observe R, S'
        Q(S,A) <- Q(S,A) + alpha [R + gamma max_a Q(S',a) - Q(S,A)]
        S <- S'
    until S is terminal
```

Double Q-learning, for estimating Q1 ~ Q2 ~ q_* (box, p.136, PDF 158; "probabilility" is the book's typo):
```
Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
Initialize Q1(s,a) and Q2(s,a), for all s in S+, a in A(s), such that Q(terminal, .) = 0
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A from S using the policy epsilon-greedy in Q1 + Q2
        Take action A, observe R, S'
        With 0.5 probabilility:
            Q1(S,A) <- Q1(S,A) + alpha ( R + gamma Q2(S', argmax_a Q1(S',a)) - Q1(S,A) )
        else:
            Q2(S,A) <- Q2(S,A) + alpha ( R + gamma Q1(S', argmax_a Q2(S',a)) - Q2(S,A) )
        S <- S'
    until S is terminal
```
Sarsa name (p.129): the update uses the quintuple (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}). "If S_{t+1} is
terminal, then Q(S_{t+1}, A_{t+1}) is defined as zero."

## 6. Equations (numbers as printed, verified on page images; S&B uses \doteq for definitions)

Eq. 6.1, constant-alpha MC (p.119, PDF 141):
```tex
V(S_t) \leftarrow V(S_t) + \alpha\Big[G_t - V(S_t)\Big]
```
"A simple every-visit Monte Carlo method suitable for nonstationary environments"; G_t = actual return after t,
alpha = constant step size "(c.f., Equation 2.4)". Named "constant-alpha MC" (p.120).

Eq. 6.2, TD(0) (p.120, PDF 142):
```tex
V(S_t) \leftarrow V(S_t) + \alpha\Big[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\Big]
```
"the target for the Monte Carlo update is G_t, whereas the target for the TD update is R_{t+1} + gamma V(S_{t+1})."
Also called "one-step TD". Gloss: R_{t+1} = reward after A_t; gamma = discount; V(S_{t+1}) = current estimate of
the next state's value (the bootstrap).

Eqs. 6.3-6.4 (p.120): what each method estimates.
```tex
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s]
         = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]
         = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]
```
(6.3) is the first line, (6.4) the last. "Monte Carlo methods use an estimate of (6.3) as a target, whereas DP
methods use an estimate of (6.4) as a target." The TD target "samples the expected values in (6.4) and it uses the
current estimate V instead of the true v_pi."

Eq. 6.5, TD error (p.121, PDF 143):
```tex
\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
```
"delta_t is the error in V(S_t), available at time t + 1."

Eq. 6.6, MC error as a sum of TD errors (p.121), valid "if the array V does not change during the episode":
```tex
G_t - V(S_t) = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(0 - 0) = \sum_{k=t}^{T-1} \gamma^{k-t}\delta_k
```
"This identity is not exact if V is updated during the episode (as it is in TD(0)), but if the step size is small
then it may still hold approximately."

Eq. 6.7, Sarsa (p.129, PDF 151):
```tex
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\Big[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\Big]
```

Eq. 6.8, Q-learning (p.131, PDF 153):
```tex
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\Big[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\Big]
```
"the learned action-value function, Q, directly approximates q_*, the optimal action-value function, independent
of the policy being followed."

Eq. 6.9, Expected Sarsa (p.133, PDF 155):
```tex
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\Big[R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A_{t+1}) \mid S_{t+1}] - Q(S_t, A_t)\Big]
            = Q(S_t, A_t) + \alpha\Big[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\Big]
```

Eq. 6.10, Double Q-learning, heads case (p.136, PDF 158):
```tex
Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha\Big[R_{t+1} + \gamma Q_2\big(S_{t+1}, \operatorname*{argmax}_a Q_1(S_{t+1}, a)\big) - Q_1(S_t, A_t)\Big]
```
"If the coin comes up tails, then the same update is done with Q1 and Q2 switched." Gloss: Q1 picks the action,
Q2 scores it, so the same noise does not both select and evaluate.

## 7. Example settings (all from the text unless marked)

Example 6.1 Driving Home (p.122, PDF 144). Rewards = elapsed time of each leg; gamma = 1; value = expected time to
go. Table as printed:

| State | Elapsed Time (minutes) | Predicted Time to Go | Predicted Total Time |
|---|---|---|---|
| leaving office, friday at 6 | 0 | 30 | 30 |
| reach car, raining | 5 | 35 | 40 |
| exiting highway | 20 | 15 | 35 |
| 2ndary road, behind truck | 30 | 10 | 40 |
| entering home street | 40 | 3 | 43 |
| arrive home | 43 | 0 | 43 |

Worked numbers in the text (p.122): exiting the highway you predicted 15 more minutes, "but in fact it took 23
minutes"; "The error, G_t - V(S_t), at this time is eight minutes. Suppose the step-size parameter, alpha, is 1/2.
Then the predicted time to go after exiting the highway would be revised upward by four minutes". Figure 6.1
(p.123): MC changes (left) and TD changes (right), both drawn for alpha = 1. Traffic-jam variant (p.122-123):
25 minutes in, predict 25 more, total 50; TD shifts the initial 30 toward 50 at once.

Example 6.2 Random Walk (boxed example, p.125, PDF 147):
- Markov reward process (no actions). Nonterminal states A, B, C, D, E in a row, terminals at both ends.
- Start: "all episodes start in the center state, C".
- Each step: left or right by one state "with equal probability".
- Reward +1 on terminating at the right; "all other rewards are zero". Undiscounted.
- True values A..E = 1/6, 2/6, 3/6, 4/6, 5/6 (v_pi(C) = 0.5).
- Initial values: "V(s) = 0.5, for all s".
- Left graph: values after 0, 1, 10, 100 episodes, single run of TD(0), alpha = 0.1 (text).
- Right graph: RMS error vs episodes, 0 to 100 "Walks / Episodes" (axis label). Alphas are printed ONLY as curve
  labels: TD alpha = .05, .1, .15; MC alpha = .01, .02, .03, .04.
- Error measure: "the root mean square (RMS) error between the value function learned and the true value
  function, averaged over the five states, then averaged over 100 runs."
- Finding (text): "The TD method was consistently better than the MC method on this task."
- Exercise 6.3 (p.126): after the first episode only V(A) changed (a self-check; the answer is not printed).
- Exercise 6.5 (p.126): at high alpha the TD RMS error "seems to go down and then up again".

Example 6.3 Random walk under batch updating (p.126-127), Figure 6.2 (p.127): after each new episode, all episodes
so far are a batch, presented repeatedly "with alpha sufficiently small that the value function converged"; RMS
error across the five states, "across 100 independent repetitions"; "the batch TD method was consistently better
than the batch Monte Carlo method."

Example 6.4 You are the Predictor (p.127-128): eight episodes: A,0,B,0 / B,1 (x6) / B,0. V(B) = 3/4. Batch TD(0)
gives V(A) = 3/4 (via the maximum-likelihood model: A -> B with 100%); batch MC gives V(A) = 0 (zero error on the
training data).

Example 6.6 Cliff Walking (p.132, PDF 154):
- "standard undiscounted, episodic task, with start and goal states", actions up, down, right, left.
- Reward -1 on all transitions except into "The Cliff"; stepping into the cliff gives -100 "and sends the agent
  instantly back to the start."
- Grid from the figure (not in the text): 4 rows x 12 columns; S bottom-left, G bottom-right, the 10 bottom-row
  cells between them are the cliff. Figure draws an "Optimal path" (one row above the cliff) and a "Safer path"
  (along the top row).
- Graph: "Sarsa and Q-learning methods with epsilon-greedy action selection, epsilon = 0.1"; x-axis Episodes 0 to
  500; y-axis "Sum of rewards during episode".
- NOT STATED for this graph: alpha, number of runs, smoothing. The page must label its own alpha as our choice.
- Findings (text): "After an initial transient, Q-learning learns values for the optimal policy, that which travels
  right along the edge of the cliff. Unfortunately, this results in its occasionally falling off the cliff because
  of the epsilon-greedy action selection. Sarsa, on the other hand, takes the action selection into account and
  learns the longer but safer path through the upper part of the grid. Although Q-learning actually learns the
  values of the optimal policy, its online performance is worse than that of Sarsa, which learns the roundabout
  policy. Of course, if epsilon were gradually reduced, then both methods would asymptotically converge to the
  optimal policy."
- Derived by us (not printed): shortest S-to-G path = 13 steps (return -13); the drawn top-row path = 17 steps.

Figure 6.3 (p.133, adapted from van Seijen et al. 2009): interim and asymptotic performance of TD control methods
on the cliff vs alpha (0.1 to 1). Caption: epsilon = 0.1; "Asymptotic performance is an average over 100,000
episodes whereas interim performance is an average over the first 100 episodes. These data are averages of over
50,000 and 10 runs for the interim and asymptotic cases respectively. The solid circles mark the best interim
performance of each method."

Example 6.7 Maximization Bias (p.134-135, Figure 6.5 p.135):
- Two nonterminal states A and B. Episodes start in A; actions left and right.
- right -> terminal, reward 0. left -> B, reward 0.
- From B: "many possible actions all of which cause immediate termination with a reward drawn from a normal
  distribution with mean -0.1 and variance 1.0" (inset: N(-0.1, 1)). The number of actions in B is NOT stated.
- So left has expected return -0.1; "taking left in state A is always a mistake."
- Parameters: epsilon = 0.1, alpha = 0.1, gamma = 1 (p.135 text); initial action values zero; ties in
  epsilon-greedy broken randomly; data averaged over 10,000 runs (caption). x-axis Episodes 1 to 300.
- Findings: "Q-learning initially learns to take the left action much more often than the right action, and always
  takes it significantly more often than the 5% minimum probability enforced by epsilon-greedy action selection
  with epsilon = 0.1. In contrast, Double Q-learning is essentially unaffected by maximization bias" (caption).
  "Even at asymptote, Q-learning takes the left action about 5% more often than is optimal at our parameter
  settings" (text).

Example 6.5 Windy Gridworld (p.130) is not used by the outline: Sarsa, epsilon = 0.1, alpha = 0.5, Q = 0; by 8000
steps the greedy policy was optimal; average episode length about 17 steps vs minimum 15.

## 8. Results and findings worth remembering
- Convergence (p.124): "For any fixed policy pi, TD(0) has been proved to converge to v_pi, in the mean for a
  constant step-size parameter if it is sufficiently small, and with probability 1 if the step-size parameter
  decreases according to the usual stochastic approximation conditions (2.7)."
- Speed (p.124): no proof either way; "In practice, however, TD methods have usually been found to converge faster
  than constant-alpha MC methods on stochastic tasks, as illustrated in Example 6.2."
- Batch optimality (p.128): "Batch Monte Carlo methods always find the estimates that minimize mean square error
  on the training set, whereas batch TD(0) always finds the estimates that would be exactly correct for the
  maximum-likelihood model of the Markov process." "In general, batch TD(0) converges to the
  certainty-equivalence estimate." Nonbatch TD "may be faster than constant-alpha MC because it is moving toward a
  better estimate" ... "At the current time nothing more definite can be said".
- Sarsa convergence (p.129): w.p. 1 to optimal "as long as all state-action pairs are visited an infinite number of
  times and the policy converges in the limit to the greedy policy (which can be arranged, for example, with
  epsilon-greedy policies by setting epsilon = 1/t)."
- Q-learning (p.131): "all that is required for correct convergence is that all pairs continue to be updated";
  "Q has been shown to converge with probability 1 to q_*."
- Expected Sarsa (p.133-134): eliminates "the variance due to the random selection of A_{t+1}"; on the cliff it
  "can safely set alpha = 1 without suffering any degradation of asymptotic performance, whereas Sarsa can only
  perform well in the long run at a small value of alpha"; "suppose pi is the greedy policy while behavior is more
  exploratory; then Expected Sarsa is exactly Q-learning."
- Maximization bias (p.134): "a maximum over estimated values is used implicitly as an estimate of the maximum
  value, which can lead to a significant positive bias." Double learning (p.135): Q2(argmax_a Q1(a)) is unbiased in
  the sense E[Q2(A*)] = q(A*); it "doubles the memory requirements, but does not increase the amount of computation
  per step." p.136: "double learning seems to eliminate the harm caused by maximization bias."
- Summary (p.138): "Sarsa is an on-policy method, and Q-learning is an off-policy method. Expected Sarsa is also an
  off-policy method as we present it here."

## 9. Limitations and history (Bibliographical and Historical Remarks, p.139-140, PDF 161-162)
- 6.1-2: "Most of the specific material from these sections is from Sutton (1988), including the TD(0) algorithm,
  the random walk example, and the term 'temporal-difference learning.'" Tabular TD(0) "was proved to converge in
  the mean by Sutton (1988) and with probability 1 by Dayan (1992), based on the work of Watkins and Dayan (1992)."
- Section 1.7 (p.21, PDF 43, Chapter 1, cited here for the TD story): "A key step was taken by Sutton (1988) by
  separating temporal-difference learning from control, treating it as a general prediction method. That paper
  also introduced the TD(lambda) algorithm". Ian Witten (1977, 1976a) "appears to be the earliest publication of a
  temporal-difference learning rule. He proposed the method that we now call tabular TD(0)". "The
  temporal-difference and optimal control threads were fully brought together in 1989 with Chris Watkins's
  development of Q-learning."
- 6.3: batch optimality of TD established by Sutton (1988); "certainty equivalence" is from adaptive control.
- 6.4: "The Sarsa algorithm was introduced by Rummery and Niranjan (1994). They explored it in conjunction with
  artificial neural networks and called it 'Modified Connectionist Q-learning'. The name 'Sarsa' was introduced by
  Sutton (1996)." Tabular Sarsa convergence proved by Singh, Jaakkola, Littman, and Szepesvari (2000).
- 6.5: "Q-learning was introduced by Watkins (1989), whose outline of a convergence proof was made rigorous by
  Watkins and Dayan (1992)." More general results: Jaakkola, Jordan, and Singh (1994); Tsitsiklis (1994).
- 6.6: Expected Sarsa introduced by George John (1994), who called it "Q-bar-learning" (Q with an overbar);
  van Seijen, van Hasselt, Whiteson, and Weiring [sic] (2009) established its convergence; Figure 6.3 is adapted from
  them; van Hasselt (2011) called the off-policy form "General Q-learning".
- 6.7: "Maximization bias and double learning were introduced and extensively investigated by van Hasselt (2010,
  2011)." The Figure 6.5 MDP is adapted from his Figure 4.1 (van Hasselt, 2011).
- Reference list entries: Rummery and Niranjan (1994), "On-line Q-learning using connectionist systems", Technical
  Report CUED/F-INFENG/TR 166, Engineering Department, Cambridge University (p.507, PDF 529). Sutton (1996),
  "Generalization in reinforcement learning: Successful examples using sparse coarse coding", NIPS 8, pp. 1038-1044
  (p.512, PDF 534). van Hasselt (2010), "Double Q-learning", NIPS 23, pp. 2613-2621 (p.515, PDF 537). Watkins
  (1989), "Learning from Delayed Rewards", PhD thesis, University of Cambridge (p.516, PDF 538).

## 10. Outline feedback (ch-mctd, ch-qlearn)
- Do not caption the random-walk learning curves "S&B Figure 6.2": in this printing they are the unnumbered right
  graph of Example 6.2 (p.125); Figure 6.2 is the batch-training result (p.127). Likewise the cliff graph is the
  unnumbered graph in Example 6.6 (p.132); Figure 6.4 is a backup-diagram figure.
- viz-walk alpha buttons: to reproduce the book's right graph use TD {0.05, 0.1, 0.15} and MC {0.01, 0.02, 0.03,
  0.04}, V init 0.5, 100 episodes, 100 runs, RMS over the 5 states. Left graph: alpha = 0.1, single run, snapshots
  at 0, 1, 10, 100 episodes.
- viz-cliff: epsilon 0.1 is sourced; alpha and runs are not. State on the page that alpha is our choice. The book
  states only qualitative findings for this graph; numeric reward levels must come from our run (do not read them
  off the book's plot).
- "Why TD wins here: bias vs variance": the book's own explanation is certainty equivalence (Section 6.3, Example
  6.4, p.127-128), and it says the general speed question is open (p.124). Example 6.4 (V(A) = 3/4 vs 0) is a
  strong, tiny worked example and self-check; consider adding it next to the bias/variance framing.
- Self-check idea from Exercise 6.3: the first episode changes only V(A); with alpha = 0.1 and V init 0.5 the new
  V(A) is 0.45 (our arithmetic, not printed).
- Worked TD numbers: the driving-home text gives sourced numbers for MC (error 8 minutes, alpha 1/2, +4 minutes).
- "Where this came from" for ch-mctd can cite Witten 1977 (earliest TD rule, S&B p.21) as well as Sutton 1988.
- ch-qlearn "Why one-step Q-learning needs no ratio": cite p.134 (greedy pi makes Expected Sarsa exactly
  Q-learning) and Section 7.5, p.152 ("Q-learning and Expected Sarsa from Chapter 6 do this for the one-step
  case"); see sb-ch07.md for the Section 7.3 explanation.
- Maximization bias: the number of actions in B is not stated; our toy must pick one and say so.
