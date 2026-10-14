---
chapter: ch-vjepa
title: V-JEPA, adding time
target_seconds: 140
---
A video is an image with a third axis. [[cue:btn-tube-reset]]
V-JEPA keeps the I-JEPA recipe and changes one thing: the hidden blocks become tubes that run through every frame.
You saw why in the last chapter.
If a block were hidden in one frame only, the cheapest prediction copies it from the next frame.
A tube removes that shortcut, so the predictor has to model what moves.

Turn the block.
Eight time slots, fourteen by fourteen tokens each, and the yellow tubes go all the way through.
This is the short-range mask: eight small blocks covering fifteen percent of the frame each. [[cue:btn-tube-short]]
And this is the long-range one: two blocks of seventy percent. [[cue:btn-tube-long]]
Each clip gets one of each, and about ninety percent of the tokens end up hidden.
Guess what hiding a single frame would let the model do, then press the ablation. [[cue:btn-tube-frame]]
Every hidden token has a visible twin one slot away.
Copy, and you are done.
That is not the paper's setting, but its random-tube and causal masks tell the same story: twenty points worse. [[cue:btn-tube-short]]

Everything else is familiar.
Sixteen frames become about fifteen hundred tokens, each covering two frames by sixteen by sixteen pixels.
The context encoder sees the visible ones, the target encoder sees the whole clip, and the predictor fills the holes with mask tokens that carry the hidden positions. [[cue:eqstep-eq-vjepa-1]]
Two changes. [[cue:eqstep-eq-vjepa-2]]
The distance is an L1, which the authors found more stable. [[cue:eqstep-eq-vjepa-3]]
And the predictor is told only where the holes are; the next chapter tells it more.

Two results to keep.
The same model trained to predict pixels instead of features drops five points on Kinetics with a frozen encoder, and the gap vanishes once everything is fine-tuned.
So feature prediction makes frozen features better; fine-tuning can repair either.
And Something-Something, a dataset where the direction of a hand motion is the label: image models score about fifty there, V-JEPA about seventy.
Its deficit on still images is the price of never having seen one, and V-JEPA 2.1 pays it off later.

One evaluation detail, because every later paper uses it.
Averaging fifteen hundred token vectors into one throws away where things are.
V-JEPA trains a one-layer attentive probe instead, a learned query that looks over the tokens, and gains seventeen points on Kinetics from the same frozen encoder.
The encoder knew; the readout had been losing it.
