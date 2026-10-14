---
chapter: ch-detr-limits
title: What it cost: long training and small objects
target_seconds: 95
---
So how did it do? [[cue:btn-cost-s32]]
On paper, a clean tie.
DETR reached 42 AP on COCO, the same as a carefully tuned Faster R-CNN, with no anchors and no NMS.

But the tie hides two very different detectors.
On large objects, DETR was much better.
On small objects, it was much worse, 20.5 against 26.6.
And it needed 500 epochs of training, where the baseline needed about 109.

Why small objects?
Look at the encoder.
Every cell of the feature map attends to every other cell, so the cost grows with the square of the number of cells.
That's why DETR works on a coarse map, one cell for every 32 by 32 pixels, where a small object might be a single cell.

Let's make the map finer. [[cue:btn-cost-s16]]
Half the stride means four times the cells, and sixteen times the pairs.
Go finer again, and it's hundreds of millions of pairs per layer. [[cue:btn-cost-s8]]
The paper tried the stride sixteen version, and speed dropped from 28 to 12 frames per second.

And the long training?
That has more than one suspect.
Attention that starts out spread almost evenly over thousands of pixels.
And object queries with no clear idea of where they are.
The rest of the story is people tackling those suspects, one at a time.
