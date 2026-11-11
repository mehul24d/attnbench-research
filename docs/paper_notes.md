# Paper notes: points the write-up must carry

A list, started 9 Nov 2026 (adviser's ruling 8 of the tenth 9 Nov 2026
paste). Each entry names a point the paper has to state, and where its
evidence is. This is not a draft and holds no new numbers. The plan
(`docs/RESEARCH_PLAN.md`) and `docs/limitations.md` win over anything here.

1. **H1's advance statement is contradicted on P1, and it was not edited.**
   H1 states in advance that the recursion and the KernelSight-style step
   form should agree on arms with no synchronisation, and that the
   recursion gains only where there are synchronisations (Section 4). On P1
   (T3.2d, the development pairing) the step form's error is lower on every
   arm, the synchronising arms included (oracle, both XAttention arms).
   P1 is never counted as a test, and H1 is scored on P2 to P7 as written.
   The paper reports the P1 result beside the statement and does not
   restate the statement after the data. Evidence: `docs/limitations.md`
   (9 Nov 2026), `docs/leg1_bootstrap_t32d_20261109.json`, Appendix A
   (9 Nov 2026).

2. **H1's list of synchronisation-free arms is contradicted by T2.3, and
   it was not edited** (adviser's ruling 1d dated 10 Nov 2026). H1 names
   dense, mean-pool and XAttention with Python asserts removed as arms "with
   no synchronisation inside the step" (Section 4). On the L4 traces of
   T3.2d only dense is: mean-pool has 2 synchronisations per layer (`bool`
   and a pinned copy in `aten::index`), XAttention with asserts removed 6
   (9 at 32768). Section 5.5's "None" for mean-pool measured the estimator
   alone. H1 is scored as written; each arm is classed from its traced
   count (`residuals.arm_class`), so no score changes. The paper states the
   statement, the traced counts and that the statement was not restated.
   Evidence: `docs/t23_l4_sync_20261110.md`, Appendix A (11 Nov 2026).
