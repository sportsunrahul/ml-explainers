# S&B Chapter 1: Introduction

## 1. Identity
- Book: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press,
  (c) 2018, 2020. File read: `sources/sutton-barto/RLbook2020.pdf` (2020 printing). First author Sutton, last
  author Barto. First edition 1998 (Preface to the Second Edition, p. xiii, PDF 13: "second edition of our 1998
  book").
- Page rule: book page = PDF page - 22 (checked against the Contents and running heads).
- Chapter 1 "Introduction", book pp. 1-22 (PDF 23-44).
- Sections: 1.1 Reinforcement Learning p.1; 1.2 Examples p.4; 1.3 Elements of Reinforcement Learning p.6;
  1.4 Limitations and Scope p.7; 1.5 An Extended Example: Tic-Tac-Toe p.8; 1.6 Summary p.13; 1.7 Early History
  of Reinforcement Learning p.13; Bibliographical Remarks p.22.

## 2. Core problem
Learn what to do (map situations to actions) to maximize a numerical reward, when nobody says which action is
right. Actions can change the next situation, and through it all later rewards. The book frames this as "the
optimal control of incompletely-known Markov decision processes" (1.1, p.2).

## 3. What prior methods did and why that hurt (per the source)
- Supervised learning learns "from a training set of labeled examples provided by a knowledgable external
  supervisor". "In interactive problems it is often impractical to obtain examples of desired behavior that are
  both correct and representative of all the situations in which the agent has to act." (1.1, p.2)
- Unsupervised learning finds hidden structure, "but by itself does not address the reinforcement learning
  problem of maximizing a reward signal." (1.1, p.2)
- Minimax (tic-tac-toe) "is not correct here because it assumes a particular way of playing by the opponent."
  Classical DP "can compute an optimal solution for any opponent, but require as input a complete specification
  of that opponent". (1.5, p.8)
- Evolutionary methods: "each policy change is made only after many games, and only the final outcome of each
  game is used: what happens during the games is ignored. ... Credit is even given to moves that never
  occurred!" (1.5, p.11)

## 4. The one key idea
Learn from evaluative interaction (trial-and-error search plus delayed reward), and use a value function to
judge individual states during play, not only whole games.

## 5. Mechanism: the tic-tac-toe value-table player (1.5, pp.9-10)
1. Keep a table with one value per board state: the estimated probability of winning. Init: 1 for states with
   three Xs in a row, 0 for three Os in a row or a full board, 0.5 for all other states (p.9).
2. To move, look up the values of the states each legal move leads to. "Most of the time we move greedily";
   "Occasionally, however, we select randomly from among the other moves instead. These are called exploratory
   moves because they cause us to experience states that we might otherwise never see." (p.9)
3. After each greedy move, "back up" the value of the later state to the earlier one with the TD update below.
   "Exploratory moves do not result in any learning" (Figure 1.1 caption, p.10).
4. "if the step-size parameter is reduced properly over time, then this method converges, for any fixed
   opponent, to the true probabilities of winning from each state given optimal play by our player" (p.10).

## 6. Equations
Chapter 1 has no numbered equations.
- TD backup, unnumbered (1.5, p.10):
  `V(S_t) \leftarrow V(S_t) + \alpha \big[ V(S_{t+1}) - V(S_t) \big]`
  Book: "This update rule is an example of a temporal-difference learning method, so called because its changes
  are based on a difference, V(S_{t+1}) - V(S_t), between estimates at two successive times."
  Gloss: V(S_t) = current estimate for the state before the greedy move; S_{t+1} = state after it; alpha = "a
  small positive fraction called the step-size parameter, which influences the rate of learning".

## 7. Settings
Tic-tac-toe table init only (step 1 above). No experiment numbers in this chapter.

## 8. Findings and quotes worth remembering
Definition (1.1, p.1, PDF 23), verbatim with em dashes as "--":
> Reinforcement learning is learning what to do--how to map situations to actions--so as to maximize a numerical
> reward signal. The learner is not told which actions to take, but instead must discover which actions yield
> the most reward by trying them. In the most interesting and challenging cases, actions may affect not only the
> immediate reward but also the next situation and, through that, all subsequent rewards. These two
> characteristics--trial-and-error search and delayed reward--are the two most important distinguishing
> features of reinforcement learning.

Exploration-exploitation (1.1, p.3, PDF 25):
> One of the challenges that arise in reinforcement learning, and not in other kinds of learning, is the
> trade-off between exploration and exploitation. ... The agent has to exploit what it has already experienced in
> order to obtain reward, but it also has to explore in order to make better action selections in the future.
> The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing at the
> task. The agent must try a variety of actions and progressively favor those that appear to be best. On a
> stochastic task, each action must be tried many times to gain a reliable estimate of its expected reward. The
> exploration-exploitation dilemma has been intensively studied by mathematicians for many decades, yet remains
> unresolved.

Same page: "the entire issue of balancing exploration and exploitation does not even arise in supervised and
unsupervised learning, at least in the purest forms of these paradigms."

Elements (1.3, pp.6-7): policy, reward signal, value function, optionally a model.
- "The agent's sole objective is to maximize the total reward it receives over the long run." (p.6)
- "Whereas the reward signal indicates what is good in an immediate sense, a value function specifies what is
  good in the long run." (p.6)
- "the most important component of almost all reinforcement learning algorithms we consider is a method for
  efficiently estimating values." (p.7)
- Model-based vs model-free: model-free methods "are explicitly trial-and-error learners--viewed as almost the
  opposite of planning." (p.7)

Games: backgammon "has approximately 10^20 states"; Tesauro's program (1992, 1995) "learned to play far better
than any previous program and eventually better than the world's best human players" (1.5, p.11).

Summary (1.6, p.13): RL "is distinguished from other computational approaches by its emphasis on learning by an
agent from direct interaction with its environment, without requiring exemplary supervision or complete models of
the environment."

### History facts for "Where this came from" (1.7, pp.13-22, unless noted)
Three threads: trial-and-error learning (from animal learning psychology), optimal control with value functions
and DP ("For the most part, this thread did not involve learning"), and temporal-difference methods. "All three
threads came together in the late 1980s to produce the modern field" (pp.13-14).

| Fact | Book wording / value | Loc |
|---|---|---|
| Law of Effect quote | "Of several responses made to the same situation, those which are accompanied or closely followed by satisfaction to the animal will, other things being equal, be more firmly connected with the situation ..." cited "(Thorndike, 1911, p. 244)" | 1.7, p.15 (PDF 37) |
| Name | "Thorndike called this the 'Law of Effect' because it describes the effect of reinforcing events on the tendency to select actions." | p.15 |
| Puzzle-box experiments | Thorndike 1898, "one hundred years before publication of the first edition of this book"; these experiments "led Thorndike to formulate a number of 'laws' of learning, the most influential being the Law of Effect" | 14.3, p.358 (PDF 380) |
| Earlier roots | trial-and-error idea back to the 1850s (Bain); Conway Lloyd Morgan's 1894 use of the term | p.15 |
| "Reinforcement" term | first in the 1927 English translation of Pavlov's monograph | p.16 |
| Turing | 1948 report, "pleasure-pain system" | p.16 |
| Optimal control | term "came into use in the late 1950s" | p.14 |
| Bellman | approach "developed in the mid-1950s by Richard Bellman and others"; value function / "optimal return function"; "a functional equation, now often called the Bellman equation"; methods "came to be known as dynamic programming (Bellman, 1957a)" | p.14 |
| MDPs | "Bellman (1957b) also introduced the discrete stochastic version of the optimal control problem known as Markov decision processes (MDPs)." | p.14 |
| Policy iteration | "Ronald Howard (1960) devised the policy iteration method for MDPs." | p.14 |
| Curse of dimensionality | DP "suffers from what Bellman called 'the curse of dimensionality,' meaning that its computational requirements grow exponentially with the number of state variables, but it is still far more efficient and more widely applicable than any other general method." | p.14 |
| Credit assignment named | Minsky (1961) "Steps Toward Artificial Intelligence": "the basic credit-assignment problem for complex reinforcement learning systems: How do you distribute credit for success among the many decisions that may have been involved in producing it? All of the methods we discuss in this book are, in a sense, directed toward solving this problem." | p.17 (PDF 39) |
| Pole balancing | Michie and Chambers (1968), BOXES, "a failure signal occurring only when the pole fell or the cart reached the end of a track"; influenced Barto, Sutton, and Anderson (1983) | p.18 |
| "Critic" | Widrow, Gupta, and Maitra (1973): "learning with a critic" instead of "learning with a teacher" | p.18 |
| Bandits | learning automata for the "k-armed bandit", Tsetlin, 1960s | p.18 |
| TD origins | "in part in animal learning psychology, in particular, in the notion of secondary reinforcers"; Minsky (1954) "may have been the first to realize" its importance for artificial learning | p.20 |
| Samuel | "Arthur Samuel (1959) was the first to propose and implement a learning method that included temporal-difference ideas, as part of his celebrated checkers-playing program" | p.20 (PDF 42) |
| Earliest TD rule | Witten (1977, 1976a) "appears to be the earliest publication of a temporal-difference learning rule"; "tabular TD(0) for use as part of an adaptive controller for solving MDPs" | p.21 |
| Actor-critic | "actor-critic architecture" applied to pole balancing (Barto, Sutton, and Anderson, 1983) | p.21 |
| Sutton 1988 | "separating temporal-difference learning from control, treating it as a general prediction method. That paper also introduced the TD(lambda) algorithm" | p.21 |
| Watkins 1989 | "The temporal-difference and optimal control threads were fully brought together in 1989 with Chris Watkins's development of Q-learning. This work extended and integrated prior work in all three threads" (p.21); "the full integration of dynamic programming methods with online learning did not occur until the work of Chris Watkins in 1989" (pp.14-15) | p.21 (PDF 43) |
| TD-Gammon | "In 1992, the remarkable success of Gerry Tesauro's backgammon playing program, TD-Gammon, brought additional attention to the field." | p.21 |
| Neuroscience | TD error vs dopamine neuron activity (Schultz, Dayan, and Montague, 1997, among others) | p.21 |

## 9. Limitations the source states
- The book does not address constructing or learning the state signal (except Section 17.3) (1.4, p.7).
- Evolutionary methods are not covered; they "ignore much of the useful structure of the reinforcement learning
  problem" (1.4, p.8).
- The exploration-exploitation dilemma "remains unresolved" (1.1, p.3).

## 10. Outline feedback (ch-score)
1. Quotes that source each step: the definition and "trial-and-error search and delayed reward" (p.1); the
   dilemma sentence (p.3); "Problem 2: the score comes late" = "delayed reward" (p.1); credit assignment is named
   by Minsky 1961 (p.17), a good anchor for Problem 2. Problem 3 (own data): "exploratory moves ... cause us to
   experience states that we might otherwise never see" (p.9). Problem 1 (score is not an answer) and the
   "reward is not a label" confusion: use the ch. 2 opening on evaluative vs instructive feedback (p.25; see
   sb-ch02.md).
2. "Where this came from": the book attributes the quoted Law of Effect to Thorndike 1911 (p.15) and the
   puzzle-box experiments to Thorndike 1898 (14.3, p.358). Do not collapse these into one year. Bellman: DP
   1957a and MDPs 1957b, both "1957"; Howard 1960 policy iteration; Watkins 1989 joins the threads. For "RL in
   games, robots": Samuel 1959 checkers, TD-Gammon 1992, Michie and Chambers 1968 pole balancing (CartPole's
   ancestor, used again in ch. 9-12). LLM post-training is not in S&B; source it from InstructGPT.
3. "The goal as one equation": J(pi) is page notation. S&B ch. 1-4 never define J(pi); J(theta) appears only in
   ch. 13 notation (Summary of Notation, p. xxi). The undiscounted return G = R_1 + ... + R_T is S&B Eq. 3.7 at
   t = 0 (sb-ch03.md); the book's goal statement is the reward hypothesis (3.2, p.53).
4. The tic-tac-toe TD backup (p.10) is unnumbered; if the page shows it here as a preview, cite "S&B 1.5", not an
   equation number.
