# Reproducibility checklist

- [ ] Python environment created from `requirements.txt`.
- [ ] SeNic h0-h29 downloaded from the public repository.
- [ ] Dataset audit reports `OK` for every participant.
- [ ] Main protocol uses session 0 only.
- [ ] Training mask is exactly p0-r0/r1.
- [ ] p0-r2 and all p1-p10 trials remain outside supervised fitting.
- [ ] Normalization statistics use training windows only.
- [ ] All RingAug models receive the same augmentation.
- [ ] Final BioAlign uses gesture cross-entropy only.
- [ ] Seeds are 42, 2026, and 3407.
- [ ] Trial probabilities, not window labels, are averaged before scoring.
- [ ] Participant, not window or seed, is the inferential unit.
- [ ] Planned angle analysis uses p0-p8 only.
- [ ] CPU timing is described as neural-forward latency, not total controller latency.
- [ ] Aggregate outputs are compared with `reference/`, allowing for environment-level numerical differences.
