# Code provenance and release status

The release was assembled from the confirmed BioAlign/BioSelect project logic and the frozen JBE manuscript protocol. It preserves the model definitions, data parsing, windowing, normalization, augmentation, subject-specific training split, trial-level aggregation, participant-level inference unit, physical-angle analysis, and CPU benchmark design used by the project.

It is a cleaned release rather than a byte-identical copy of the original workstation directory. Changes made for release quality include:

1. replacing the hard-coded `D:\\BioSelect_EMG` root with command-line paths;
2. separating reusable code into the `bioalign_emg` package;
3. adding explicit validation and error messages;
4. adding smoke tests, manifests, and SHA-256 checksums;
5. documenting the legacy registered-parameter count versus the active forward path;
6. excluding public raw data, caches, and trained checkpoints.

The aggregate values in `reference/` are transcription targets from the frozen manuscript and must not be treated as newly generated output. A successful reproduction requires running the code on the public SeNic files.
