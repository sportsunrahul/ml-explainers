# blalock2020 -- What is the State of Neural Network Pruning? (BRANCH, analysis)

## Identity
- Title: What is the State of Neural Network Pruning?
- arXiv 2003.03033, read v1 (2020-03-06, only version) (abs.html).
- Authors: Davis Blalock ... John Guttag (arXiv listing). Venue: MLSys 2020 (abs.html comments).
- Repo named in the paper: https://github.com/jjgo/shrinkbench (ShrinkBench).

## Key idea (two sentences)
A meta-analysis of 81 pruning papers finds that results cannot be compared: few papers compare to each other,
they use different (dataset, architecture) pairs and inconsistently defined metrics. Its clearest finding is
"a lack of standardized benchmarks and metrics"; it proposes ShrinkBench to fix this.

## Key numbers (Sec. 3.1, 4.1-4.3, 5.2; `\npapers` macro = 81, doc.tex)
- 81 papers surveyed; results use 49 datasets, 132 architectures, 195 (dataset, architecture) combinations.
- Most common pair, VGG-16 on ImageNet, appears in only 22 of 81 papers (Table 1, Sec. 4.2).
- Only 37 of 81 report any result on the four most common configurations (Fig. 3 caption).
- More than a fourth compare to no prior pruning method; another fourth compare to only one; nearly all compare
  to three or fewer (Sec. 4.1, Fig. 2).
- Reported AlexNet ImageNet FLOPs vary up to a factor of four: 371, 724 and 1500 MFLOPs (Sec. 5.2).
- Sec. 2.3: unstructured pruning "may not be arranged in a fashion conducive to speedups using modern libraries
  and hardware"; Sec. 2.4 (metrics): "Parameter and FLOP counts are a loose proxy for real-world latency".
- Sec. 3.2 / Fig. 1: pruning "generally does not help as much as switching to a better architecture".

## Outline feedback
- Use Blalock for the critique (no standard benchmark; FLOPs and params are proxies; "speedup" is defined
  inconsistently), not as evidence of GPU slowness: the paper measures no hardware latency.
- The "pruning vs better architecture" finding (Fig. 1) supports the guide's ordering (distill or swap backbone
  before unstructured pruning) (our reading).
