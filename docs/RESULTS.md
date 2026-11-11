# Results and limitations

## Final held-out study (completed)

The stored CSVs were extracted directly from 20 frozen evaluation reports: **10 independent training seeds per MAPPO controller variant**, 8 evaluation episodes per controller/method, and 10 allocation methods. The exports include **1,600 method-level episode returns** and original model/scenario/protocol hashes for audit.

| Controller variant | Switch-Aware | Original R-CART | R-CART − Switch-Aware |
|---|---:|---:|---:|
| Fault-robust (predeclared primary) | **93.540** | 91.418 | **−2.123** |
| Nominal (secondary) | 94.763 | **95.160** | +0.398 |

The reported fault-robust 95% **training-seed bootstrap interval** for the paired primary return difference was approximately **[−5.160, +0.352]**. Because it crosses zero, superiority is **not established**; the point estimate itself disfavors original R-CART. It would be misleading to describe this as an R-CART win.

The completed primary study also found R-CART suffered large paired losses for fault-robust seeds 202 (−12.300) and 404 (−9.225). Follow-up post-hoc trace analysis identified one harmful decision mechanism: the raw synchronization wait term can penalize a near-rover coalition twice, favoring distant agents despite equal forecast task value. This is a **diagnosis after the final test**, not an independently confirmed fix.

## Development-only comparisons

Under the earlier three development patterns, the original R-CART had fault-robust seed-level average deltas relative to Switch-Aware of **−0.488** (single failure), **+1.868** (repeated failures), and **+2.933** (capability changes / delayed failures). These were useful development findings, not confirmatory results.

A later two-seed Service-Aware development pilot tied Switch-Aware in all 24 paired episodes. This shows the revised planner ran successfully and avoided the observed regression in those cases, **but did not establish a distinct, beneficial assignment mechanism**.

Ranking CFT matched analytic R-CART in most development episodes. Predictive error reductions alone are not evidence of better role allocation.

## Demo vs. experiment

The **historical six-rover test** included an earlier, *not bundled* short
animation with one fault-robust controller (training seed 202), development
failure schedule, and episode seed 900000. Its Switch-Aware and original
R-CART returns were **120.0 and 118.8**, respectively. These historical
values are **not from either twelve-rover video** currently available under
[`media/`](../media/) and are not evidence of superiority.

The **two published twelve-rover demos** use training seed 314159 and the
same evaluation seed 901110. Switch-Aware versus Fixed returns
**786.7 / 639.5**; Switch-Aware versus original R-CART returns
**786.7 / 786.7**. These are illustrative trajectories of one frozen actor,
not independent-seed inferential evidence. The complete eight-episode
allocator comparison and its limitations are documented in
[`RESULTS_12_ROVERS.md`](RESULTS_12_ROVERS.md).

## Replication caveats

The source includes code and exported actor-only examples. Reproducing the full study's *training* requires retraining the independently seeded controllers or obtaining the original full training checkpoints (not bundled due to size). The published CSVs and provenance provide independently inspectable episode returns but are not a replacement for raw training and optimizer-state artifacts. The old frozen study's path names were normalized for a clean public repository; the hashes in `results/manifest.json` refer to the **original saved run**. Never falsely compare a renamed/reinitialized protocol's checksum to the archived one.
