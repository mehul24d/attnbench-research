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
