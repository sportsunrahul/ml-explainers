# Resume notes (2026-10-09, session stopped at the usage limit)

## Done
- Checkpoint 0 approved (see state.json). Sources for 17 spine + 5 edge papers in sources/; 33 branch papers fetched by a background job (check notes/branch-fetch.log for DONE).
- outline.md, notes/extract-brief.md, notes/design.md, notes/map-data.json, notes/claims/map.json.
- Finished extraction: patchcore (344 claims), efficientad (339), spade-padim (154). Six spine agents were still running at the stop: benchmarks, flows-synthetic, uniad-ader, vlm, dinov2, edge. Check notes/papers/ and notes/claims/ for their files; re-run any group whose files are missing.
- Branch extraction agents were NOT started (two planned: precursors/feature/flows/synthetic/ST; unified/zero-few-shot/benchmarks). Output: notes/branches.json + notes/claims/branches.json.
- Page skeleton: explainer.head.part, parts/body-top.html, parts/lib-toy.js (toy world, simulated in build/toy-*.mjs), parts/viz-hero.js, parts/ch-problem.html, parts/viz-normal-cloud.js, build.sh. bash build.sh PASSES check-page.

## Fix next (from screenshots in build/figs)
- Hero: labels collide at the panel's top-left ("normal features" vs "test patch"); on the cap the scratch lands only at the cloud's edge (2D gap 0.90 vs spread ~5). Switch the hero to the mesh (tight cloud, scratch 2D gap ~1.3) or move the label and zoom.
- ch-problem: works (bank 1 -> false alarm, 20 -> pass, scratch -> reject). The flat line of dots at the bottom is background patches; mention it in the caption or drop background patches.

## Outline corrections from the finished agents
- PatchCore: random projection is only for coreset selection (bank keeps 1024-d); Eq. 7 reweighting is not in the official code (plain max, k = 1); layers are upsampled and average-pooled to 1024, not concatenated; no VisA numbers; hardware printed "Nvidia Tesla V4", 0.17 s per image for PatchCore-1%.
- EfficientAD: Table 1 accuracy = mean over MVTec AD, VisA, LOCO (use Table 2 MAD column for MVTec-only); Table 17 prints latency on A6000, A5000, V100, 3080, 2080 Ti; p_hard ablation in Table 3 for the slider; add AST 2210.07829 as a branch; some claim values are words ("twelve", "five").
- SPADE: pixel score uses kappa = 1 (nearest single feature); image score = mean distance to K = 50 nearest images with pooled layer-4 features; use SPADE's own pixel AUROC 96.0.
- PaDiM: image score = max of Mahalanobis map; "550 of 1792" is our reading; storage 3.8 GB (Table VIII); add the Rd-MVTec rotation/crop step (PaDiM -5.3, SPADE -8.8); CPU-only timings (SPADE 7.10 s, PaDiM 0.95 s).
- check-page lint: some claims carry "our_reading": true; teach the lint or keep them uncertain.

## Then
understanding.md -> fresh fact-check agent -> checkpoint 2 (outline + chapter 1 on the long page; atlas stage 1 preview).
Ask the user to save the MVTec LOCO PDF (Springer, CC BY, browser challenge) to sources/mvtec-loco/paper.pdf.
