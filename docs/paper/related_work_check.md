# Related work: existence check (28 Nov 2026)

The paper-quality wave of 28 Nov 2026, item 6. Each reference in the professor brief's list (`docs/brief/brief.html`, Section 4: 31 entries and 5 "distant" ones) and SpargeAttn (`docs/related_work_verified.md`, entry 4) was looked up on 28 Nov 2026. The lookups used arXiv abstract pages and the arXiv API, Crossref, ACL Anthology and PMLR. dblp, OpenAlex, Semantic Scholar and the AAAI OJS PDF were blocked or rate-limited. No identifier below comes from memory.

**This checks existence and identifiers only.** It does not make a paper "read in full" for rule 5 (CLAUDE.md) or task T7.2; read-in-full status is the brief's and `docs/related_work_verified.md`'s. `paper/refs.bib` is still empty, as rule 5 requires.

## Verified (title plus an arXiv ID or DOI on a fetched page)

| # | Paper (first author) | Identifier as fetched | Venue as fetched | Note |
|---|---|---|---|---|
| 1 | A Holistic Model for End-to-End Latency Prediction of Deep Learning Tasks on Heterogeneous Systems, "CTFusion" (Gennai) | DOI 10.1109/flta67013.2025.11336244 | IEEE FLTA 2025 | Cite by full title: arXiv:2605.11504 is an unrelated "CTFusion" |
| 2 | TaxBreak (Vellaisamy) | arXiv:2603.12465; DOI 10.1109/ispass69572.2026.00014 | IEEE ISPASS 2026, pp. 28–38 | The DOI is new: the 28 Oct file found none |
| 3 | KernelSight-LM (Yao) | arXiv:2606.28565 (v2) | arXiv only | |
| 4 | APEX: Asynchronous Parallel CPU-GPU Execution… (Fan) | DOI 10.1109/ipdps65963.2026.00099; arXiv:2506.03296 | IEEE IPDPS 2026 | |
| 5 | nn-Meter (Zhang) | DOI 10.1145/3458864.3467882 | ACM MobiSys 2021 | Not the 2022 GetMobile article |
| 6 | HELP (Lee) | arXiv:2106.08630 | arXiv comment: NeurIPS 2021 | |
| 7 | CloserToMe (Tang) | DOI 10.1609/aaai.v40i30.39779 | AAAI 2026, vol. 40(30) | |
| 8 | LLMServingSim 2.0 (Cho) | arXiv:2602.23036 | **ISPASS 2026 not confirmed** | Existence verified; venue unconfirmed (Crossref rate-limited) |
| 9 | VSPrefill (Chen) | arXiv:2603.04460 | arXiv only | |
| 10 | Less Is More (Yang) | arXiv:2508.07101 | arXiv only | |
| 11 | CompactAttention (Song) | arXiv:2605.16839 | arXiv comment: NeurIPS 2026 | |
| 12 | Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures (Vellaisamy) | arXiv:2504.11750; DOI 10.1109/ISPASS64960.2025.00015 (arXiv's related-DOI link) | ISPASS 2025 | |
| 13 | Vidur (Agrawal) | arXiv:2405.05465 | **MLSys 2024 not shown** | |
| 14 | Hardware-Assisted Cross-Generation Prediction of GPUs Under Design (O'Neal) | DOI 10.1109/tcad.2018.2834398 | IEEE TCAD 38(6), 2019 | |
| 15 | The Framework Tax (Fernandez) | arXiv:2302.06117 | arXiv comment: EMNLP 2023 | |
| 16 | The Sparse Frontier (Nawrot) | arXiv:2504.17768; DOI 10.18653/v1/2026.findings-acl.1926 | Findings of ACL 2026 | The venue is now confirmed: the brief's "reported, not shown" can be updated. ACL lists the second author as "Jianing Li", arXiv as "Robert Li" |
| 17 | MInference 1.0 (Jiang) | arXiv:2407.02490 | arXiv comment: NeurIPS 2024 | |
| 18 | XAttention (Xu) | arXiv:2503.16428; PMLR 267:69819–69831 | ICML 2025 (PMLR page) | |
| 19 | DuoAttention (Xiao) | arXiv:2410.10819 | **ICLR 2025 not shown** | |
| 20 | FlexPrefill (Lai) | arXiv:2502.20766 | arXiv comment: ICLR 2025 | |
| 21 | Twilight (Lin) | arXiv:2502.02770 | arXiv comment: NeurIPS 2025 | |
| 22 | Improving the Performance of LLM Inference: Understanding the Role of CUDA Graphs (Franco) | DOI 10.1109/lanman69841.2026.11623563 | IEEE LANMAN 2026 | |
| 24 | iMLBench (Zhang) | DOI 10.1109/tpds.2020.3046870 | IEEE TPDS 32(7), 2021 | |
| 25 | Producing Wrong Data Without Doing Anything Obviously Wrong! (Mytkowicz) | DOI 10.1145/1508244.1508275 | ASPLOS 2009 | |
| 26 | How Much Dense Attention is Necessary? (Wang) | arXiv:2606.07703 | arXiv technical report | |
| 27 | AB-Sparse (Liu) | arXiv:2605.12110 | arXiv only | |
| 28 | Token Sparse Attention (Jo) | arXiv:2602.03216 | arXiv comment: ICML 2026 | |
| 29 | APEX: An Extensible and Dynamism-Aware Simulator… (Lin) | arXiv:2411.17651 | arXiv only | |
| 30 | HiSparse (Xie) | arXiv:2608.07009 | arXiv only | |
| 31 | DFSAttn (Hu) | arXiv:2605.23445 | arXiv comment: ICML 2026 | The brief gives no ID; this one was found by title |
| — | SpargeAttention (Zhang) | arXiv:2502.18137 | ICML 2025 (PMLR 267, journal-ref) | |
| distant | HyGIN (Huang) | DOI 10.1109/lca.2026.3711652 | IEEE CAL 25(2), 2026 | **Description mismatch:** the title says Mixture-of-Experts training |
| distant | Exploratory Study of CPU–GPU Frequency Interactions… in GPU-Accelerated 5G O-RAN (Elango) | DOI 10.1109/infocom59046.2026.11571401 | IEEE INFOCOM 2026 | The brief's "5G stack" is "5G O-RAN" |
| distant | LLMEdger: Phase-Aware Model Parallelism Scheduler for LLM Inference on Edge (Shen) | DOI 10.1109/icfec69006.2026.00010 | IEEE ICFEC 2026 | **Description unconfirmed:** the record has no abstract and does not mention encoder-decoder or CPU-only nodes |

## Not verified: drop until confirmed

| # | Paper as the brief gives it | Why |
|---|---|---|
| 23 | I/O-Aware PIM Acceleration for Long-Sequence LLM Inference with Hybrid Sparse Attention, "PILOT" (Lu), IEEE IPDPS 2026 | No fetched page showed it. The arXiv title search returned 0; Crossref was rate-limited (429) on two attempts |
| distant | Kohli et al., IEEE HPEC 2025 (CPU, GPU and NPU benchmarking on one edge chip) | Crossref returned no Kohli or HPEC 2025 match; the brief gives no title |
| distant | "PLA", IEEE TCAD 2026 (token-pruning accelerator for point-cloud LLMs) | arXiv returned 0; Crossref was rate-limited |

Since the brief says the researcher supplied the three "distant" entries, they may well exist. They are dropped from any citable list only until a DOI is fetched. The researcher's own copies would settle it.

**Other flags.**
- Three venues are not shown on any page fetched here: LLMServingSim 2.0 (ISPASS 2026), Vidur (MLSys 2024) and DuoAttention (ICLR 2025). Recheck each in IEEE Xplore, Scopus or the proceedings before submission (task T7.3).
