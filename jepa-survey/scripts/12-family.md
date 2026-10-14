---
chapter: ch-family
title: One idea, every modality
target_seconds: 110
---
The recipe never mentions pixels. [[cue:btn-tree-reset]]
Define a context and a target, encode both, predict the target's embedding from the context's, stop it collapsing.
Every branch on this tree changes only what the blocks are.

Here is the main line: the position paper, I-JEPA, V-JEPA, V-JEPA 2 and 2.1, LeJEPA, LeWorldModel, H-JEPA. [[cue:btn-tree-spine]]
And here is everything else. [[cue:btn-tree-all]]
Audio: spectrogram patches with a mask that learns time and frequency.
Point clouds: a sequencer that orders the points so block masking can be reused.
Graphs: targets compressed onto a hyperbola so the space carries hierarchy.
Language: keep the next-token loss and add a JEPA term between a description and its code.
Vision and language together: predict the embedding of the target text from video, and decode only when text is needed.
Generation: add a diffusion loss on the masked tokens and the architecture becomes an image generator.
Objects: mask object slots instead of patches, so interactions must be modelled.
And a variational version that makes encoder and predictor probabilistic in one objective.

Two cautions as you read the table.
MC-JEPA, despite the name, is not a masked JEPA at all; it is VICReg multi-tasked with optical flow.
And the EB-JEPA library is not a method but a teaching codebase, worth knowing because it ablates every regulariser: remove the inverse-dynamics term and planning success drops from ninety-seven to one.
Select any node to read its delta card.
