---
chapter: ch-deformable
title: Deformable DETR: look at a few points, not every pixel
target_seconds: 100
---
Here's the idea that fixed the attention. [[cue:btn-deform-reset]]
Does a query really need to compare itself against every single pixel?
Deformable DETR says no.
Give the query a reference point, the yellow dot, and let it pick a handful of places nearby to look.

Each of eight heads samples four points.
And there are no query-key dot products at all.
The query itself predicts where to look, and how much to trust each look.

This is how the points start out, straight from the paper's initialization. [[cue:btn-deform-init]]
Eight directions, like a compass, and each head's points march outward, one step at a time.
Try dragging the yellow dot around the image.

Now add the other feature scales. [[cue:btn-deform-multi]]
The same pattern lands on four maps, from fine to coarse, so the query sees both detail and context.
That's 128 points per query, no matter how big the image is.

After training, the points move to where the information is. [[cue:btn-deform-trained]]
This part is a toy, but it matches what the paper describes: the samples spread over the object and its edges.

So the cost now grows in a straight line with the image, not with its square.
And attention is focused from the very start, instead of smeared across thousands of pixels.
The result was dramatic.
43.8 AP in 50 epochs, against DETR's 42 in 500.
And small objects jumped from 20.5 to 26.4.
Two extras mattered for what came next: refining the box layer by layer, and starting queries from the encoder's best guesses.
