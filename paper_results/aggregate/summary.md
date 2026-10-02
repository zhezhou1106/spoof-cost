# Multi-seed regime summary

Design: shared paper E0 + three E2 seeds (Probe / SCL / Cut). No `probe_seq`.
Seeds: [0, 1, 2]
Paper E0: results/e0
Regime ordering stable (none/probe/scl/cprobe): False

| Arm | Regime agreement | Adoption steps | Hack median [min,max] | Honest median [min,max] |
|-----|------------------|----------------|-----------------------|-------------------------|
| none | ['unknown'] / agree=False | [None] | 0.984375 [0.984375, 0.984375] | 0.0 [0.0, 0.0] |
| probe | ['adopted', 'adopted', 'adopted'] / agree=True | [28, 58, 49] | 0.9609375 [0.953125, 0.96875] | 0.0078125 [0.0078125, 0.0078125] |
| scl | ['not_reached', 'adopted', 'adopted'] / agree=False | [None, 40, 96] | 0.8828125 [0.15625, 0.9375] | 0.03125 [0.0078125, 0.421875] |
| cprobe | ['adopted', 'not_reached', 'not_reached'] / agree=False | [54, None, None] | 0.1328125 [0.09375, 0.9296875] | 0.3046875 [0.0078125, 0.34375] |
