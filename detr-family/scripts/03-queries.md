---
chapter: ch-queries
title: Object queries and a decoder that talks to itself
target_seconds: 95
---
Where do those 100 slots come from? [[cue:btn-dec-reset]]
The architecture is almost boring, and that's the point.
A ResNet turns the image into a coarse grid of features.
A transformer encoder lets every grid cell look at every other cell.
Then a decoder reads that grid through 100 learned vectors called object queries, and each query becomes one slot.

The decoder has two kinds of attention.
Cross-attention lets each query read the image.
Self-attention lets the queries read each other.
And that second one is how they coordinate.
If two queries drift toward the same car, self-attention gives them a way to notice, and the matching loss gives one of them a reason to back off.

Here's a toy version of what happens, layer by layer. [[cue:btn-dec-l1]]
In the first layer there are several confident boxes on each object.
Now watch them settle. [[cue:btn-dec-play]]
By the last layer, each object keeps one box, and the rest fade to "no object".

The paper checks this for real.
It runs old-fashioned NMS on each layer's output.
NMS helps the first layer, but by the last layers it stops helping, and even hurts a little.
From the first decoder layer to the last, accuracy goes up by 8.2 AP.

Hold on to one detail.
These object queries are just learned vectors.
Nothing in them says "I'm a box, over here".
The next few papers are largely about what a query should really be.
