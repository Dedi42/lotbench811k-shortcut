# Detector codes

The score files and the code use short codes; the paper uses names.

| Code | Name in the paper | View |
|---|---|---|
| B1 | Hotelling *T*² (chamber means, shrinkage covariance) | one chamber off |
| B2 | Chamber deviation | one chamber off |
| B3-L1 | Lot-summary lasso | lot as a whole |
| B3-RF | Lot-summary forest (reference model) | lot as a whole |
| B3-GB | Lot-summary GBDT (gradient boosting, default settings; revision 2) | lot as a whole |
| B4 | Wafer classifier (gradient boosting, top-3 wafers) | some wafers bad |
| B5 | Isolation forest | unlike normal lots |
| B6 | Deep one-class | unlike normal lots |
| B7 | Deep Sets | set of wafers |
| B8 | Set Transformer | set of wafers |
| B9 | Attention MIL | set of wafers |
| B10 | Tool commonality (one-sided Fisher test over route steps) | suspicious tool on the route |
| SALE | Structure-aware lot encoder | all views in one network |
| SALE-S | Structured SALE | hand-built structure + forest |
| SALE-lite | SALE without the components that did not help in the ablations | |
| MCE | Stacked ensemble (mechanism-complementary ensemble) | combination |
| MCE-lite, MCE-without-SALE, MCE-without-SALE-and-SALE-S | MCE variants of Section 5.2 | |
| TI-RF, TI-NN, GB-U, PU-RF, SSL-RF | variants reported only in Online Resource 1 | |
| CTX | context-only model (shortcut test, not a candidate) | |

Fault scenarios on LotBench-811K: F1 faulty chamber, F2 post-maintenance seasoning, F3 within-lot drift,
F4 upstream tool in a time window, F5 material batch, F6 intermittent event; N = normal lot.
F4 and F5 are visible in route or context by design.
