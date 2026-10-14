---
chapter: ch-anchor-queries
title: DAB-DETR: a query is a box
target_seconds: 100
---
Remember that loose end? [[cue:btn-anchor-reset]]
What is an object query, really?
DAB-DETR splits it in two.
There's a content part, what the query has seen so far.
And there's a positional part, where it's looking.
In DETR, the positional part is an opaque learned vector, and its attention wanders over several unrelated places at once.

So make the positional part literally a box.
An x, a y, a width and a height.
The red glow is where this box's positional attention goes.
Its center decides where to look, and its size decides how wide to look.

Watch what happens without the size. [[cue:btn-anchor-mod-off]]
The attention is the same shape, whatever the box.
Now switch it back on. [[cue:btn-anchor-mod-on]]
A wide box spreads the attention sideways, a tall box stretches it up and down.
Try the width and height sliders yourself.

There's a small surprise in the encoding, too.
The standard sine encoding uses a temperature of ten thousand, which was designed for word positions in a sentence. [[cue:btn-anchor-t10000]]
For box coordinates between zero and one, that makes attention almost flat.
DAB-DETR uses twenty instead. [[cue:btn-anchor-t20]]

And after every decoder layer, the box gets nudged toward the object. [[cue:btn-anchor-refine]]
Six small corrections, and it lands.
In the paper, removing just that layer-by-layer update costs 1.7 AP.
Because the query is a box, the decoder always knows where, and how big, to look.
