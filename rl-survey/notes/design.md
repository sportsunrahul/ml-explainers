# Design notes: "Learning from a score"

Subject: reinforcement learning taught from scratch as a course. Audience: an ML engineer who knows supervised
deep learning and is new to RL. Job: make each algorithm click by running it in front of the reader, then hand
them code that ran.

## Color (house tokens; one meaning each, whole page)

| Token | Dark | Light | Meaning |
|---|---|---|---|
| --blue | #58c4dd | #1f6f9a | value estimates: V, Q, the critic |
| --yellow | #f4d35e | #8a6700 | rewards and returns: the score |
| --green | #83c167 | #3f7d20 | the policy and its actions: the actor, arrows, chosen moves |
| --red | #fc6255 | #c0392b | errors and losses: TD error, gradient, loss |
| --brown | #c8906d | #8b5a2b | fixed, not learned: walls, cliffs, hazards, frozen networks, a stored model |

Neutrals are the house tokens (--bg, --fg, --muted, --rule, --panel, --panel2). Code is grayscale on purpose:
keywords bold, comments muted italic, strings and numbers in --fg. Color never means two things on this page.

## Gridworld visual language (yard, cliff, maze, hero)

- Cells: rounded squares on --bg inside the figure panel, 2 px gap. A value fills the cell with --blue at an opacity
  proportional to (V - Vmin) / (Vmax - Vmin), clamped to .06-.85; the number sits in the cell in --fg.
- Terminal reward cells: a --yellow ring with the reward printed (+1, -1, -100). Hazards (stairwell, cliff) also get
  --brown diagonal hatching: fixed parts of the world. Walls: solid --brown.
- Policy: --green chevrons in each cell (length = probability for stochastic policies).
- The agent: a --fg disc with a small heading tick; its trail fades over the last few steps.
- An update flashes the updated cell's border --red for 300 ms (the TD error that moved it).

## Type

House: Charter body 1.125rem/1.7; headings in KaTeX_Main (embedded with the math CSS, no extra bytes); UI text in
system-ui; code in ui-monospace, SFMono-Regular, Menlo, Consolas at .86rem/1.55.

## Layout

```
| rail |  h2 Chapter title                         (40rem prose measure, left aligned)
|      |  [Exploration] [Credit] [Your own data]   problem chips: filled = this chapter
|      |  two-sentence setup
|      |  +------------------------------------------------+  figure stage, 60rem
|      |  | canvas (live sim)                       badge |
|      |  | controls: Run  Step  Reset  speed --o--       |
|      |  | readout (aria-live)                           |
|      |  +------------------------------------------------+
|      |  h3 steps ... equation (stage width) ... gloss ... worked numbers
|      |  details.check  .recap  .aside.practice
|      |  +-- Code it yourself ----------------------------+  stage width
|      |  | [NumPy] [PyTorch]                    Copy     |
|      |  | pre.code ...                                   |
|      |  | Output from our run                            |
|      |  | Try this: ...                                  |
|      |  +------------------------------------------------+
```

## Components

- Problem chips: three pills after each h2 (exploration, credit assignment, your own data). Filled --fg pill = the
  chapter works on it; outlined --muted pill = not this chapter. Not numbered: the three are a set, not a sequence.
- Notation card: a `details` under the color key with the S&B notation table, and the r_t vs R_{t+1} warning.
- Exercise: `div.exercise` with a tablist (NumPy, PyTorch), one `pre.code` per tab inlined at build time from
  `code/`, its real output in `pre.code-out`, and a "Try this" list. A Copy button per tab.
- Live sims: Run / Step / Reset plus a speed slider where the run is long. Runs are seeded; Reset restores the seed,
  so a reader sees the same run twice. Animation only while the reader runs it.
- Recorded runs: a scrubber (range input) over checkpoints with playback of the recorded episode.

## Principles

1. The figure runs the algorithm. A live figure is the real update rule on a small world, not an animation of one.
2. Color is meaning. Code stays grayscale so a color never means two things.
3. One motion moment: the hero. Everything else moves only when the reader acts.
4. Numbers you can check: readouts show the same numbers the worked examples use.

## Rejected defaults

Rainbow syntax highlighting (would reuse the meaning colors); uppercase eyebrows over sections; a card grid of
chapters; fade-in on scroll; numbered markers on things that are not sequences; gradient washes and panel shadows.
