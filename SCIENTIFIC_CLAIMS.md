# Scientific claim boundary

## What the legacy V1 results support

The legacy experiment supports a **classification** claim: a learned cyclic transformation module improved trial-level Macro-F1 under the SeNic shift protocol relative to the included neural baselines.

## What the legacy V1 results do not establish

They do not by themselves establish that:

1. the latent distribution estimates physical armband rotation;
2. the operation reconstructs the original electrode-to-muscle map;
3. soft weighting preserves rather than blurs gesture-specific spatial structure;
4. an eight-bin integer roll is an adequate model of measured displacement;
5. the gain exceeds classical, explicit-correction, oracle, or topology-matched controls.

## Evidence required before using the phrase “rotation repair”

The V2 experiment must show all of the following:

- recovery of known synthetic integer and fractional shifts;
- agreement with measured real displacement, with circular error statistics;
- improved same-gesture cross-position feature similarity after alignment;
- no loss of gesture separability after alignment;
- better performance than uniform, random, shuffled, hard, continuous, and no-alignment controls;
- comparison with explicit circular cross-correlation and oracle correction;
- classical LDA/SVM and topology-aware neural baselines;
- participant-level two-sided inference and uncertainty intervals.

Until then, use **latent cyclic transformation** or **shift-robust classification**, not **physical rotation estimation** or **electrode-map restoration**.
