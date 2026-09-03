# Registered versus active BioAlign parameters

The frozen project defined the final `BioAlignEMG` by subclassing the exploratory `BioSelectEMG` and overriding `forward()`. The inherited `ring_gate` and `time_gate` objects therefore remained registered in the module even though the final forward pass did not call them.

Consequences:

- PyTorch registered/trainable parameter count: **33,549**.
- Parameters on the final computational forward path: **31,299**.
- Difference: **2,250** parameters in the two unused registered gate modules.

The release keeps this layout in `BioAlignEMG` so original state dictionaries can be loaded and the manuscript-reported registered parameter count can be reproduced. `BioAlignEMGCompact` removes the two unused modules and reports 31,299 parameters, but it must be retrained before numerical equivalence can be claimed.

For a public manuscript/code release, the authors should decide between two transparent choices:

1. preserve the 33,549-parameter checkpoint-compatible model and explicitly state that this is the registered parameter count; or
2. retrain the compact 31,299-parameter model and update the parameter count, model-size benchmark, and any affected results.
