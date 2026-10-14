---
chapter: ch-matching
title: One prediction per object: Hungarian matching
target_seconds: 100
---
So how does DETR output a set? [[cue:btn-match-reset]]
It predicts a fixed number of slots, 100 of them, all at once.
Each slot says a class and a box, and one of the classes is "no object".
Most images have only a handful of things in them, so most slots should say exactly that.

But during training, there's a question we have to answer first.
Which slot is responsible for which object?
Here's the trick.
Pad the list of real objects with "no object" entries until it's as long as the list of slots.
Then find the one-to-one pairing that costs the least.

Look at the grid.
Each row is a predicted slot, each column is a real object or a "no object" pad.
A cell is the cost of pairing that slot with that column: how unsure the slot is about the right class, plus how far its box is from the true box.
Try picking one cell per row, and see if you can beat the optimum.

Here's the answer. [[cue:btn-match-optimal]]
Notice slot two. [[cue:btn-match-dup]]
It's a perfectly decent box on the car, but slot one fits the car better, and each object can only be claimed once.
So slot two gets paired with "no object", and the loss trains it to stay silent.

That's the whole idea.
Duplicates become expensive during training, so you never need to clean them up afterwards.

Here's the same idea in symbols. [[cue:eqstep-eq-assign-1]]
We search over every way of pairing slots with objects, and keep the cheapest. [[cue:eqstep-eq-assign-2]]
Each pairing compares a real object, in yellow, with a slot's prediction, in blue. [[cue:eqstep-eq-assign-3]]
The cost has two parts: how unsure the slot is about the right class, [[cue:eqstep-eq-cost-1]]
and how far its box is from the true box. [[cue:eqstep-eq-cost-2]]
Once the pairing is fixed, every slot is trained toward its class, [[cue:eqstep-eq-loss-1]]
and the slots that found a real object are trained toward its box too. [[cue:eqstep-eq-loss-2]]
