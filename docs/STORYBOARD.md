# Storyboard: first body of work

The regression suite is a small exhibition. Each scene is chosen so that at
least one style has a strong tradition for it, and each tests a different
failure mode.

| # | Scene | Why it is in the set | Stress test |
|---|-------|----------------------|-------------|
| 1 | Harbour at dusk: sun, boat, sea, distant hills | classic poster subject | horizon, reflection, overprint |
| 2 | Mountain village: mountains, 3 houses, pine trees | depth by layering | many small objects, clutter |
| 3 | Still life: vase, 3 fruits, table | indoor, light direction | form without photoreal shading |
| 4 | Moonrise: full moon, stars, pine ridge, lake, rowboat | vortex skies | impasto flow, dark palettes (composition kept away from any famous night painting) |
| 5 | Bamboo & bird | sumi-e canon | economy, empty paper |
| 6 | Lighthouse on cliff with sea | tall thin subject | cropping, silhouette |
| 7 | Single tree in field with sun | minimal scene | nearly-empty canvas must not be "blank" |
| 8 | Sailboats (two) and gulls | small repeated objects | recognisability at small size |
| 9 | Teapot and cup | object identity | "helmet that looks like a phone" risk |
| 10 | Fishing boat in rain on river | weather | texture vs. clutter |

Held-out scenes (never iterated on by the generator loop) live in
`evals/holdout/`: a windmill under clouds, pears beside a cup, a moonlit lake. They share the object
vocabulary but use new combinations and layouts.

## Production order

1. `core/` primitives, `scene/` spec and shapes, screenprint style,
   harbour scene end to end.
2. Loop (render, look, critique, revise) on the harbour.
3. Technical checks + suite.
4. Judges, pairwise Elo, HTML report.
5. Sumi-e and impasto built in parallel against `STYLE_GUIDE.md`, then
   compared with the screenprint baseline on the suite.
