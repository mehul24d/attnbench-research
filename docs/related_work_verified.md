# Related work: full-text verification (task C9, 28 Oct 2026)

This file checks ten key references against their full text and against the way `docs/RESEARCH_PLAN.md` describes them. Plan lines are cited by line number at commit `9d2d8b1` (branch `cloud/C5-2026-10-28`). The plan is not edited here. Each mismatch is a proposal for the researcher, not a change.

## How this was checked

- Each PDF was downloaded from `arxiv.org/pdf/<id>` (latest version, named below) into a scratch directory outside the repository and not committed. Text was extracted with `pypdf`, page by page. The arXiv abstract page was read for authors, version history, comments and DOI.
- "PDF p. N" is the page index in the arXiv PDF, which can differ from a printed page number.
- Quotes are copied from the extracted text. Mechanical repairs only: words split by a line-end hyphen are rejoined ("estima-tion" becomes "estimation"), and the hyphen is kept where the word itself is hyphenated ("trace-driven"); spaces the extractor dropped between words are restored ("validateTaxBreakon" becomes "validate TaxBreak on"); curly apostrophes are written straight; single math symbols inside a sentence are written in plain text (t_gpu, c_ovl, c_tail, P). Equations are transcribed from the rendered page and labelled as transcriptions, not quotes. Every quote was checked by script against the extracted text with whitespace and hyphens ignored.
- The full text of each paper was obtained and searched; the cited passages were read in context. This is not a claim that every page of every paper was read line by line. Whether these checks count as "read in full" for Rule 5 and task T7.2 is the researcher's decision.
- CTFusion could not be read here (Section 6).

## Summary

| # | Paper | Read here | Plan wording matches the paper? |
|---|---|---|---|
| 1 | XAttention | Yes (arXiv v1) | Yes on method. ICML 2025 is not stated in the arXiv PDF or record (see flags). |
| 2 | MInference 1.0 | Yes (arXiv v2) | Yes for the plan. The brief's "5% to 30%" drops the condition the paper attaches to it. |
| 3 | FlexPrefill | Yes (arXiv v1) | Head-adaptive: yes. "Entropy-driven": the paper uses Jensen-Shannon divergence; the word "entropy" does not occur. |
| 4 | SpargeAttn | Yes (arXiv v8) | Not cited anywhere in the plan, the brief or the repository. Nothing to match. |
| 5 | TaxBreak | Yes (arXiv v1) | Yes, including the null-kernel protocol, Eq. 3, the 4.7 us floor, the "aggregate residual" remark and the replay caveat. |
| 6 | KernelSight-LM | Yes (arXiv v2) | Mostly. Three wording issues (Eq. 4 has an extra term; the paper contradicts itself on GPU dependence; "does not transfer" is stronger than the paper). |
| 7 | CTFusion | **Unreadable here** (not on arXiv; needs `ieeexplore.ieee.org`, blocked) | Not verified. |
| 8 | CompactAttention | Yes (arXiv v2) | Yes. 0.68x to 0.84x reproduces from its Tables 3 and 4. Venue is now NeurIPS 2026 (the brief says preprint). |
| 9 | SKIP (Vellaisamy et al. 2025) | Yes (arXiv v1) | Yes on TKLQT and 2.37 us. "SKIP" is the profiler's name, not the paper's. Venue is ISPASS 2025 (the brief says preprint). |
| 10 | The Sparse Frontier | Yes (arXiv v3) | Close. The paper's stated unknown is narrower than the plan's G3 wording. |

---

## 1. XAttention

- **Title:** XAttention: Block Sparse Attention with Antidiagonal Scoring
- **Authors:** Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, Song Han
- **Venue, year:** ICML 2025 (see flag 1). arXiv record: no journal reference; comment "The first two authors contributed equally to this work".
- **arXiv:** 2503.16428 (v1, 20 Mar 2025, the only version)

**How the plan characterises it.** Line 224: "Antidiagonal scoring with per-(layer, head) thresholds from the authors' profiler, run as shipped. XAttention: Xu et al., "XAttention: Block Sparse Attention with Antidiagonal Scoring", ICML 2025; arXiv:2503.16428". Line 478 (T7.2) lists XAttention among "the brief entries known from abstracts only".

**Quotes.**
1. Abstract, PDF p. 1: "XAttention's key innovation is the insight that the sum of antidiagonal values (i.e., from the lower-left to upper-right) in the attention matrix provides a powerful proxy for block importance."
2. Section 2.2 (Threshold Block selection), PDF p. 3: "Finally, for block selection, the find blocks function is employed to identify the minimal set of blocks whose cumulative sum of antidiagonal probabilities exceeds a predefined threshold τ."
3. Section 2.3 (Minimum Threshold Prediction), PDF p. 3: "We propose a dynamic programming approach to determine the optimal threshold for each attention head." The same section ends: "Note that this dynamic threshold prediction method can further optimize XAttention's sparsity but is not a mandatory component."

**Flags.**
1. *Venue not shown in the paper.* The arXiv v1 PDF carries no venue line, and the arXiv record has no journal reference. The only ICML evidence found is the citation block in the code repository's README (`mit-han-lab/x-attention`), which gives `booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)}`. A second attempt on 28 Oct 2026 to confirm it from the proceedings (`proceedings.mlr.press/v267/` and `icml.cc`) failed: both hosts are blocked from this container. The venue is therefore **unconfirmed** here. The plan's "ICML 2025" is plausible but rests on the authors' README, not the paper or the proceedings.
2. *Per-(layer, head).* The paper says "for each attention head", over H heads. That is consistent with the plan if H counts heads across all layers; the paper does not say "layer" in this section. Not a mismatch, but the plan's "(layer, head)" is the repository's granularity, not a phrase from the paper.
3. *Optional component.* The paper calls the threshold search "not a mandatory component". The plan's "Calibrated XAttention" arm uses it. No conflict, but worth stating when the arm is described.
4. *Code claims are not paper claims.* Plan line 33 ("XAttention runs two `assert` checks on GPU tensors per query chunk") and line 224 (`find_blocks_chunked`) describe upstream code at `e379887`. The paper says nothing about them; they are not checked here.
5. *T7.2.* Line 478 still lists XAttention as abstract-only. This check supplies full-text quotes; updating the row is the researcher's call.

## 2. MInference 1.0

- **Title:** MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
- **Authors:** Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu
- **Venue, year:** NeurIPS 2024. PDF p. 1 footer: "38th Conference on Neural Information Processing Systems (NeurIPS 2024)." arXiv comment: "Accepted at NeurIPS 2024 (Spotlight)".
- **arXiv:** 2407.02490 (v2, 30 Oct 2024)

**How the plan characterises it.** Line 222: "Mean-pool | Pooled query and key blocks (MInference-style); inline device builder". Line 223: "Vertical-slash style | Vertical and slash scoring reduced to blocks". Line 478 lists MInference as abstract-only. The brief (reference 17) adds: "Source of the mean-pool and vertical-slash estimators; reports 5% to 30% of time spent building indices on one A100."

**Quotes.**
1. Section 1, PDF p. 2: "For Block-Sparse heads, we perform mean pooling on both query and key vectors in blocks of 64 and calculate the block-level attention weights to determine the most important blocks and thereby obtain a block-sparse dynamic mask."
2. Section 1, PDF p. 2: "For example, to build a dynamic sparse mask for a specific prompt on one Vertical-Slash head, we use a partial of attention weight consisting of the last last_q query and key vectors (i.e. Q[−last_q:] and K) to estimate the most important indices of the vertical and slash lines globally on the attention matrix."
3. Appendix A (Limitations), PDF p. 18: "As the context length decreases, the time required to build the dynamic index becomes more significant as attention computation time decreases. For example, with a 10k context, the time spent on building the index increases from 5% to 30%, resulting in overall end-to-end latency approaching that of FlashAttention."

**Flags.**
1. *Plan:* no mismatch. Quote 1 supports "pooled query and key blocks"; quote 2 supports the vertical-slash estimator.
2. *Brief, not plan:* "5% to 30%" is the paper's figure for a 10k context as context shrinks (quote 3). Elsewhere the paper gives other ranges. Section 4, PDF p. 9: "we found about 5%-20% of the overhead is spent on dynamic sparse index building". Appendix, PDF p. 22: "The estimation and index-building time for the dynamic sparse pattern accounts for approximately 5%-15% and 25% of the total time for Vertical-Slash and Block-Sparse patterns, respectively." Proposed brief wording: "reports index building rising to 30% of time at a 10k context on one A100". The plan itself does not cite the figure.
3. *T7.2.* Line 478 still lists MInference as abstract-only.

## 3. FlexPrefill

- **Title:** FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference
- **Authors:** Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, Xun Zhou
- **Venue, year:** ICLR 2025. PDF p. 1 header: "Published as a conference paper at ICLR 2025". arXiv comment: "Accepted at ICLR 2025 (Oral)".
- **arXiv:** 2502.20766 (v1, 28 Feb 2025)

**How the plan characterises it.** Line 514 (declined directions): "A head-adaptive or entropy-driven estimator as the contribution | Already explored (FlexPrefill, Twilight, AB-Sparse)". Line 33: "the FlexPrefill implementation bundled with it sizes index tensors with `.max().item()`". Line 478 lists FlexPrefill as abstract-only. Brief reference 20: "Adapts sparsity per head and prompt".

**Quotes.**
1. Abstract, PDF p. 1: "FlexPrefill adaptively optimizes the sparse pattern and sparse ratio of each attention head based on the prompt, enhancing efficiency in long-sequence inference tasks."
2. Abstract, PDF p. 1: "1) Query-Aware Sparse Pattern Determination: By measuring Jensen-Shannon divergence, this component adaptively switches between query-specific diverse attention patterns and predefined attention patterns. 2) Cumulative-Attention Based Index Selection: This component dynamically selects query-key indexes to be computed based on different attention patterns, ensuring the sum of attention scores meets a predefined threshold."
3. Section 1, PDF p. 2: "This approach allows for the adaptive allocation of computational budgets for different attention heads while maintaining the effectiveness of the model."

**Flags.**
1. *"Entropy-driven".* The paper's switch is a Jensen-Shannon divergence test (Section 3, PDF p. 4: "We adopt the Jensen-Shannon divergence for its symmetry and boundedness properties"); the word "entropy" does not appear in the extracted text. Line 514's "head-adaptive or entropy-driven" is still true of FlexPrefill through "head-adaptive". If FlexPrefill is meant as the entropy example, "divergence-driven" is more exact.
2. *Code claim.* Line 33's `.max().item()` describes the implementation bundled with XAttention, not the paper. Not checked here.
3. *T7.2.* Line 478 still lists FlexPrefill as abstract-only.

## 4. SpargeAttn

- **Title:** SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference (arXiv v8 title; the authors' BibTeX in the arXiv comment uses "Spargeattn: Accurate sparse attention accelerating any model inference")
- **Authors:** Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, Jianfei Chen
- **Venue, year:** ICML 2025. PDF p. 1: "Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025." arXiv journal reference: "Proceedings of the 42 nd International Conference on Machine Learning, PMLR 267, 2025 (ICML 2025)".
- **arXiv:** 2502.18137 (v8, 19 Nov 2025)

**How the plan characterises it.** It does not. "Sparge" occurs nowhere in the plan, the brief or any other file in the repository.

**Quotes** (what the paper says about itself, for the researcher's decision):
1. Abstract, PDF p. 1: "Our method uses a two-stage online filter: in the first stage, we rapidly and accurately predict the attention map, enabling the skip of some matrix multiplications in attention. In the second stage, we design an online softmax-aware filter that incurs no extra overhead and further skips some matrix multiplications."
2. Section 1, PDF p. 2: "This is because accuracy demands precise prediction of the sparse regions in the attention map, while efficiency requires the overhead of this prediction to be minimal."
3. Section 4.3, PDF p. 7, with Table 3 on PDF p. 8: "Table 3 compares the overhead of dynamic sparse block prediction in SpargeAttn compared with attention execution latency. The results indicate that the prediction overhead is minimal". Table 3 lists overhead from "8k 0.251 6.649 3.78%" down to "128k 8.764 1696.2 0.516%".

**Flags.**
1. *Researcher's decision (relayed 28 Oct 2026):* SpargeAttn joins T7.2 at integration, as "claims verified against full text (machine-checked quotes)". No plan or brief text describes it, so there is nothing to mismatch. The one bibliographic wrinkle: the arXiv v8 title ("SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference") differs from the title in the authors' BibTeX ("Spargeattn: Accurate sparse attention accelerating any model inference"); cite the PMLR title once it can be checked against the proceedings, which are blocked here.
2. *Placement.* If the plan should cite it, it fits Section 3 as another training-free estimator. Its overhead is reported against the attention kernel alone (Table 3 compares prediction time with "Full Attention (ms)"), not end to end, which matches the plan's premise that kernel-level figures are what papers report. Section 3.6 (PDF p. 5) states that the "method incorporates three hyper-parameters" set per attention layer, so it is training-free but calibrated, like XAttention's threshold search. Adding it is a scope decision for the researcher.

## 5. TaxBreak

- **Title:** TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition
- **Authors:** Prabhu Vellaisamy, Shreesh Tripathi, Vignesh Natarajan, Surya Santhan Thenarasu, Shawn Blanton, John P. Shen
- **Venue, year:** IEEE ISPASS 2026. arXiv comment: "Accepted at IEEE ISPASS 2026. Copyright assigned to IEEE". No DOI on the arXiv record.
- **arXiv:** 2603.12465 (v1, 12 Mar 2026)

**How the plan characterises it.** Line 89 (G2): "TaxBreak splits host-visible time into framework, library and launch-path parts, and we adapt its null-kernel protocol to measure host costs (null kernels across the whole operation sequence rather than the launch path alone); it does not compose those costs in execution order with synchronisation positions ... (TaxBreak's remark that host cost is often shown "only as an aggregate residual" describes earlier work, not TaxBreak.) ... TaxBreak says its replay can be imperfect for synchronisation-heavy kernels." Line 262: "TaxBreak about 4.7 us on Xeon 8480C with H100". Line 637: "HDBI is sum d / (sum d + sum h), TaxBreak's Eq. 3 (arXiv:2603.12465)".

**Quotes.**
1. Abstract, PDF p. 1: "Existing approaches typically expose this cost only as an aggregate residual or a launch/queue metric, which is often insufficient to identify which execution layer should be optimized. This work presents TaxBreak, a trace-driven methodology for decomposing host-visible orchestration overhead into three components: framework translation time, CUDA library translation time, and kernel launch-path time." Section II.D, PDF p. 3, names the earlier work: "Existing studies either present host-side overhead as an aggregate residual [14] or localize it mainly to the H2D (host to device) launch path (TKLQT) [30]." Reference [14] is Fernandez et al., The Framework Tax; [30] is the SKIP paper (Section 9).
2. Section III.B, PDF p. 4: "Phase 2 first measures the dynamic system floor with a null-kernel nsys run, then replays each unique kernel in isolation to measure dispatch translation, launch-floor cost, and their residuals without queue interference." Section III, PDF p. 3, defines the launch term as "the hardware floor from cudaLaunchKernel call to GPU kernel start, measured via null-kernel profiling."
3. Section VII, PDF p. 10: "Also, replay-based attribution can be imperfect for highly dynamic, autotuned, or synchronization-heavy kernels."

Supporting the 4.7 us figure: Section IV.A, PDF p. 5: "H100 platform: Intel Xeon 8480C (PCIe Gen5, 56 cores @ 2.0/3.8 GHz) paired with an NVIDIA H100 (80 GB) GPU in a DGX H100 system." Table III, PDF p. 7, row "H100 4.707 4.578 4.260 5.396" (avg, p50, p5, p95, in us); Section V.A, PDF p. 6: "with a per-kernel floor of about 4.7µs (Table III)". Eq. 3, transcribed: HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestration), where T_Orchestration (Eq. 2) is a sum over all kernel invocations.

**Flags.** None on substance. The plan's description of the protocol, the remark about earlier work, Eq. 3, the 4.7 us floor and the replay caveat all match. Note for line 262: TaxBreak's floor is from an `nsys` null-kernel run on a DGX H100 (H100 80 GB, SXM form factor implied by DGX), which matters when it is set beside SKIP's PCIe figure (Section 9, flag 3).

## 6. KernelSight-LM

- **Title:** KernelSight-LM: A Kernel-Level LLM Inference Simulator
- **Authors:** Xiteng Yao, Taeho Kim, Hengzhi Pei, Xinle Liu, Kyle Ulrich, Leonard Lausen, Ashish Khetan, Xiang Song, George Karypis, Martin Herbordt
- **Venue, year:** arXiv preprint, 2026 (PDF running header "arXiv Preprint, June, 2026"; no venue on the record)
- **arXiv:** 2606.28565 (v2, 2 Jul 2026)

**How the plan characterises it.** Line 88 (G1): "KernelSight-LM measures host constants per host and assumes they do not depend on the GPU, without testing it." Line 89 (G2): "composes within a step by one formula, GPU time plus dispatch, plus any excess of the host overlap floor over GPU time, plus a serial tail (its Section 5.7, Eq. 4); its discrete-event scheduler advances one event per batch step (Section 4.2), so nothing records where in a step a synchronisation falls, and its synchronisation term is tensor-parallel all-reduce between GPUs. Its Tier B measures host constants on the target with a short serving sweep." Line 518: "KernelSight-LM finds attention-kernel efficiency does not transfer; device costs are measured".

**Quotes.**
1. Section 5.7, PDF p. 7: "The step wall time is not simply t_gpu, because the serving engine overlaps host work (launch, sampling, scheduling) with GPU compute". After Eq. 4: "where the host launch/overlap floor c_ovl surfaces only when it exceeds the GPU and the serial tail c_tail is always on the critical path." Eq. 4, transcribed from the rendered page: t_step = max(t_gpu + n·c_disp, ℓ) + max(0, c_ovl − t_gpu) + c_tail.
2. Section 4.2, PDF p. 4: "When a batch is ready for execution, the simulator invokes our kernel-level execution time predictor: given the batch composition, the predictor aggregates individual kernel latencies into a total batch execution time. This predicted duration is then used to schedule an event to advance the simulation clock by the predicted execution time."
3. Section 5.7, PDF p. 7: "These overhead constants depend on the host CPU and serving engine, not on the GPU, so we measure them once from a short serving sweep during Tier B characterization (§5.5)". Section 5.6, PDF p. 6: "Tensor-parallel inference inserts an all-reduce after the attention and MLP blocks of each layer to combine partial activations across the P shards".

**Flags.**
1. *Eq. 4 has a term the plan's paraphrase drops.* The first term is max(t_gpu + n·c_disp, ℓ), not t_gpu + n·c_disp. The symbol ℓ is not defined anywhere in the extracted text. Line 89's "GPU time plus dispatch" is accurate only when that sum exceeds ℓ. Proposed wording: "the larger of GPU time plus dispatch and a floor ℓ, plus any excess ...". The plan's own step form (line 197) is its own construction and is unaffected.
2. *The paper contradicts itself on GPU dependence.* Section 5.7 (quote 3) says the constants depend "not on the GPU". Section 5.5, PDF p. 6, says the opposite: "Being properties of the (host CPU, GPU, engine) stack rather than the GPU alone, they are re-measured whenever the host or serving engine changes". Section 1, PDF p. 2: "We measure host-side overhead terms—CPU/GPU overlap, scheduling tail, admission, launch, and TP synchronization—per (host, GPU, engine)". Line 88's "assumes they do not depend on the GPU" is supported by Section 5.7, but a reader of Section 5.5 could object. Proposed: cite Section 5.7 explicitly and note that Section 5.5 describes them as properties of the host, GPU and engine together while re-measuring only when the host or engine changes.
3. *"Does not transfer" is stronger than the paper.* Section 8, PDF p. 11: "the main open problem is the limited cross-generation transfer of the attention efficiency factor, which motivates one-shot per-architecture attention calibration". Section 7.3, PDF p. 10: "attention is the least-transferable kernel". Appendix, PDF p. 15: "closing the gap requires at least one serving-matched measurement on each new architecture rather than pure extrapolation." Proposed wording for line 518: "finds attention-kernel efficiency transfers poorly across GPU generations".
4. Line 89's other claims (one event per batch step; synchronisation term is tensor-parallel all-reduce; Tier B short serving sweep) match quotes 2 and 3.

## 7. CTFusion

- **Status: unreadable here.** Not on arXiv. An arXiv title search ("Holistic Model End-to-End Latency Prediction Heterogeneous") and an author search ("Gennai Ohsita") returned no results. An arXiv search for "CTFusion" returns only an unrelated paper with the same name (arXiv:2605.11504, "CTFusion: A CTF-based Benchmark for LLM Agent Evaluation", Lee, Bae and Yun), so the name alone is ambiguous in citations. The paper needs `ieeexplore.ieee.org` (IEEE FLTA 2025), which is blocked from this container.
- **Bibliographic data, from the brief only (unverified):** "A Holistic Model for End-to-End Latency Prediction of Deep Learning Tasks on Heterogeneous Systems", Gennai, Ohsita and Shimonishi, IEEE FLTA 2025. No DOI recorded in the repository.

**How the plan characterises it.** Line 40: "Cross-device latency prediction itself is not new (CTFusion)". Line 88: "Learned cross-device predictors exist (CTFusion: 20 training systems, host side is input preprocessing)." Line 198: "Learned (CTFusion-style LightGBM)".

**Quotes.** None; no text available here.

**Flags.** None of these claims could be checked: the 20 training systems, the preprocessing host side, the 17.6% error (brief), and whether CTFusion uses LightGBM specifically or gradient-boosted trees in general (the brief says "gradient-boosted trees"; the plan says "LightGBM"). It stays unreadable until the researcher supplies the PDF through their institutional IEEE access (researcher's decision, relayed 28 Oct 2026). Cite with the full title, since "CTFusion" alone collides with arXiv:2605.11504.

## 8. CompactAttention

- **Title:** CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection
- **Authors:** Jiwon Song, Dongwon Jo, Beomseok Kang, Jae-Joon Kim
- **Venue, year:** NeurIPS 2026. PDF p. 1 footer: "40th Conference on Neural Information Processing Systems (NeurIPS 2026)." arXiv comment: "NeurIPS 2026 Spotlight".
- **arXiv:** 2605.16839 (v2, 19 Oct 2026)

**How the plan characterises it.** Line 287: "External reference only: CompactAttention reports XAttention at 0.68x to 0.84x of dense speed end to end under chunked prefill on two GPUs." Line 90 (G3): "AB-Sparse and CompactAttention compare accuracy at fixed budgets or per-method operating points." Line 516: "Benchmarking estimator cost against quality | Method papers and CompactAttention already report it".

**Quotes.**
1. Section 5 (Environment), PDF p. 7: "We measure attention latency on two NVIDIA GPUs." The Figure 5 caption, PDF p. 7, gives the settings: "(a) RTX PRO 6000 GPUs with TP=2, batch size 4, and chunk size 512. (b) H200 SXM GPUs with TP=2, batch size 8, and chunk size 1024."
2. Section 5, PDF p. 7: "XAttention and SeerAttention are often slower than dense attention, reflecting repeated pattern search overhead and inefficient block-sparse execution in the Q≪KV regime." Figure 1 caption, PDF p. 2: "(c) Pattern search cost accumulates across chunks, with XAttention incurring the highest overhead."
3. Section 5 (setup), PDF p. 6: "For all other sparse methods, we set sparsity hyperparameters independently for each method and model to select accuracy-preserving operating points."

**Check of 0.68x to 0.84x.** Computed here as dense end-to-end latency divided by XAttention end-to-end latency, from Appendix Tables 3 and 4 (PDF p. 14), 8K to 128K context. RTX PRO 6000: 0.80, 0.78, 0.83, 0.82, 0.84. H200: 0.74, 0.70, 0.68, 0.70, 0.68. The range 0.68 to 0.84 holds. These ratios are this file's arithmetic on the paper's tables, not a sentence in the paper.

**Flags.**
1. "On two GPUs" means two GPU types, each run with tensor parallelism 2 (four devices per setting). Clearer: "on two GPU types (RTX PRO 6000 and H200, each with TP=2), on one model".
2. *Brief, not plan:* reference 11 lists it as "arXiv preprint (2605.16839)". The v2 PDF and arXiv record now state NeurIPS 2026.

## 9. SKIP (TKLQT)

- **Title:** Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures
- **Authors:** Prabhu Vellaisamy, Thomas Labonte, Sourav Chakraborty, Matt Turner, Samantika Sury, John Paul Shen
- **Venue, year:** IEEE ISPASS 2025. arXiv comment: "Accepted for ISPASS 2025". DOI: 10.1109/ISPASS64960.2025.00015.
- **arXiv:** 2504.11750 (v1, 16 Apr 2025)

**How the plan characterises it.** Line 89 (G2): "Host-overhead summaries (TaxBreak's HDBI, SKIP's TKLQT) are ratios or sums." Line 262: "compared only with a published figure measured the same way (TaxBreak about 4.7 us on Xeon 8480C with H100; SKIP 2.37 us on Xeon 8468V with H100 PCIe)".

**Quotes.**
1. Section I, PDF p. 2: "To obtain these insights, we developed SKIP (System-Aware Kernel Inference Profiler), a PyTorch-based profiler that builds operator-kernel dependency graphs from PyTorch Profiler [1] traces to benchmark LLM inference on state-of-the-art hardware."
2. Section III.A, PDF p. 5: "where n is the total number of kernels launched. TKLQT directly measures the accumulated overhead associated with the CPU-GPU kernel offload process, making it highly sensitive to CPU limitations (launch cost) and GPU saturation (queuing)." TKLQT (Eq. 2) is the sum over kernels of t_l = t_sb(k) − t_sb(l), the begin time of the kernel minus the begin time of its launch call (Eq. 1).
3. Table IV, PDF p. 7: "2. 2P Intel Xeon Platinum 8468V (48-core) with 512GB, H100 PCIe (350W): Intel+H100". Table V, PDF p. 8 ("CudaLaunchAPI with nullKernel launch overhead and nullKernel duration across the evaluation platforms"): "Intel+H100 2374.6 1235.2" (ns).

**Flags.**
1. "SKIP" is the paper's profiler, not its title. Fine as shorthand, but the reference should be the title, authors and ISPASS 2025 DOI above.
2. *Brief, not plan:* reference 12 lists it as "arXiv preprint (2504.11750)"; the arXiv record gives ISPASS 2025 with a DOI. The brief also calls the host "Sapphire Rapids"; the SKIP paper names the CPU only as "Intel Xeon Platinum 8468V" (TaxBreak, not SKIP, uses "Sapphire Rapids", for the 8480C).
3. *"Measured the same way" (line 262).* 2374.6 ns rounds to 2.37 us, so the figure is right. But SKIP's number comes from a PyTorch-Profiler-based tool (quote 1), and TaxBreak's from an `nsys` null-kernel run on a DGX H100 (Section 5). The two published floors differ by about 2x on Sapphire Rapids hosts with H100s. Whichever is used as the comparison for P4 should be named with its instrument and definition, not only its value.

## 10. The Sparse Frontier

- **Title:** The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- **Authors:** Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio, Edoardo M. Ponti
- **Venue, year:** arXiv preprint, latest version 2026. No venue in the v3 PDF or on the arXiv record (no comment, no journal reference). The brief's "ACL Findings 2026 reported, not shown in the paper" is confirmed as not shown.
- **arXiv:** 2504.17768 (v3, 22 Jun 2026)

**How the plan characterises it.** Line 90 (G3): "Whether recall is a sufficient accuracy proxy across estimators is untested. The Sparse Frontier lists it as unknown." Line 517: "An attention-mass oracle is a reference, not a task optimum (oracle-guided sparse prefill; The Sparse Frontier)". Brief reference 16: "Reports FLOPs, not wall-clock time".

**Quotes.**
1. Limitations ("Sixth"), PDF p. 10: "Sixth, the methods we study estimate importance from attention scores, implicitly targeting attention-recall maximisation rather than downstream task performance. How close a recall-maximising mask is to a task-loss-optimal one remains unknown; measuring this gap—ideally while ignoring estimation and kernel-efficiency constraints to isolate the pattern itself—would quantify how much headroom remains beyond better estimators and could motivate alternative selection criteria."
2. Limitations ("Third"), PDF p. 10: "Third, we report hardware-agnostic computational costs (FLOPs and memory access) rather than wall-clock timings. Actual speedups depend on hardware, batch size, and implementation quality, which vary across deployment environments."
3. Abstract, PDF p. 1: "2) for the training-free methods we study, fine-grained per-query importance estimation during prefilling remains impractical—due to both the cost of estimation and the lack of sparse kernels that translate fine-grained sparsity into wall-clock gains".

**Flags.**
1. *G3 wording (line 90).* The paper's stated unknown is how close a recall-maximising mask is to a task-loss-optimal mask (quote 1). The plan's "it" is whether recall is a sufficient accuracy proxy across estimators. The two are related, but the paper does not mention comparing estimators at matched recall. The brief's wording ("names as unknown how close a recall-maximising mask is to a task-optimal one, which is the G3 question") is closer. Proposed for line 90: "The Sparse Frontier lists the gap between recall-maximising and task-optimal masks as unknown."
2. Line 517 and the brief's FLOPs remark match quotes 1 and 2.

---

## Edits for integration

Researcher's decisions relayed on 28 Oct 2026. These edits are made at integration, not on this branch: this branch does not edit the plan or the brief. Each entry quotes the current text (plan at `9d2d8b1`) and gives the replacement. "us" follows the plan's spelling of microseconds.

### Plan

**P1. Line 89 (G2), KernelSight-LM's Eq. 4.**
Current: "KernelSight-LM (read in full) composes within a step by one formula, GPU time plus dispatch, plus any excess of the host overlap floor over GPU time, plus a serial tail (its Section 5.7, Eq. 4);"
Replacement: "KernelSight-LM (read in full) composes within a step by one formula: the larger of GPU time plus dispatch and a term ℓ that the paper does not define, plus any excess of the host overlap floor over GPU time, plus a serial tail (its Section 5.7, Eq. 4);"

**P2. Line 197, the KernelSight-style step form.**
Current: "- **KernelSight-style step form:** max(sum h, sum l + sum d) per layer, plus every synchronisation cost as a serial tail."
Replacement: "- **KernelSight-style step form:** our construction from KernelSight-LM's Eq. 4, without the term ℓ the authors leave undefined: max(sum h, sum l + sum d) per layer, plus every synchronisation cost as a serial tail."

**P3. Line 88 (G1), KernelSight-LM's host constants.**
Current: "KernelSight-LM measures host constants per host and assumes they do not depend on the GPU, without testing it."
Replacement: "KernelSight-LM measures host constants per host and states that they \"depend on the host CPU and serving engine, not on the GPU\" (its Section 5.7), without testing it; its Section 5.5 calls them properties of the host, GPU and engine together, yet re-measures them only when the host or serving engine changes."

**P4. Line 518, declined directions.**
Current: "KernelSight-LM finds attention-kernel efficiency does not transfer; device costs are measured"
Replacement: "KernelSight-LM finds attention-kernel efficiency transfers poorly across GPU generations (\"limited cross-generation transfer\", its Section 8); device costs are measured"

**P5. Line 262, null-kernel floor plausibility.**
Current: "compared only with a published figure measured the same way (TaxBreak about 4.7 us on Xeon 8480C with H100; SKIP 2.37 us on Xeon 8468V with H100 PCIe)"
Replacement: "compared only with a published figure measured the same way, named with its tool: TaxBreak, 4.7 us, the mean null-kernel launch floor from an `nsys` null-kernel run on a Xeon 8480C with an H100 in a DGX H100 (its Table III); SKIP, the PyTorch-Profiler-based profiler of Vellaisamy et al. (ISPASS 2025), 2.37 us, null-kernel CUDA launch-API overhead on a Xeon Platinum 8468V with an H100 PCIe (its Table V)"

**P6. Line 90 (G3), The Sparse Frontier.**
Current: "Whether recall is a sufficient accuracy proxy across estimators is untested. The Sparse Frontier lists it as unknown."
Replacement: "Whether recall is a sufficient accuracy proxy across estimators is untested. The Sparse Frontier lists a narrower question as unknown: \"How close a recall-maximising mask is to a task-loss-optimal one\" (its Limitations)."

**P7. Line 478 (T7.2), status.**
Current (status cell): "In progress (finished: DFSAttn and HiSparse, read in full 5 Oct 2026; remaining: Janus and MOMO, and the brief entries known from abstracts only: nn-Meter, HELP, Vidur, the cross-generation GPU prediction paper (TCAD 2019), The Framework Tax, MInference, XAttention, DuoAttention, FlexPrefill, Twilight, the CUDA Graphs study (LANMAN 2026), PILOT, iMLBench, Mytkowicz et al. and Token Sparse Attention)"
Replacement: "In progress (finished: DFSAttn and HiSparse, read in full 5 Oct 2026; claims verified against full text (machine-checked quotes, `docs/related_work_verified.md`, 28 Oct 2026): MInference, FlexPrefill, SpargeAttn; to be read in full by the researcher: XAttention, KernelSight-LM, TaxBreak, CTFusion (PDF via IEEE access); remaining: Janus and MOMO, and the brief entries known from abstracts only: nn-Meter, HELP, Vidur, the cross-generation GPU prediction paper (TCAD 2019), The Framework Tax, DuoAttention, Twilight, the CUDA Graphs study (LANMAN 2026), PILOT, iMLBench, Mytkowicz et al. and Token Sparse Attention)"
A dated Appendix A line records the decision.

### Brief (`docs/brief/brief.html`; rebuild `docs/research_brief.pdf` afterwards)

**B1. Section 4, gap table (G1 cell).**
Current: "KernelSight-LM assumes its host costs do not depend on the GPU but does not test this."
Replacement: "KernelSight-LM states that its host costs depend on the host and serving engine, not on the GPU, but does not test this."

**B2. Section 4, gap table (G2 cell), Eq. 4.**
Current: "combines host and GPU time inside a step with one formula: GPU time, plus any excess of host time over it, plus a fixed tail (its Section 5.7, Eq. 4)."
Replacement: "combines host and GPU time inside a step with one formula: the larger of GPU time plus dispatch and a term the paper does not define, plus any excess of host time over GPU time, plus a fixed tail (its Section 5.7, Eq. 4)."

**B3. Reference 3 (KernelSight-LM).**
Current: "and the host constants are assumed not to depend on the GPU. Our KernelSight-style step form, one of the baselines, is built from its step formula; it is our construction, not their simulator."
Replacement: "and the host constants are said to depend on the host and engine, not the GPU (its Section 5.7). Our KernelSight-style step form, one of the baselines, is built from its step formula without the term the authors leave undefined; it is our construction, not their simulator."

**B4. Reference 11 (CompactAttention), venue and setting.**
Current venue cell: "arXiv preprint (2605.16839)". Replacement: "NeurIPS (arXiv:2605.16839)".
Current: "Under chunked prefill on two GPUs, XAttention's end-to-end time is 0.68x to 0.84x of dense speed". Replacement: "Under chunked prefill on two GPU types (RTX PRO 6000 and H200, two of each), XAttention's end-to-end time is 0.68x to 0.84x of dense speed".

**B5. Reference 12 (Vellaisamy et al., SKIP), venue and host.**
Current venue cell: "arXiv preprint (2504.11750)". Replacement: "IEEE ISPASS (arXiv:2504.11750; DOI 10.1109/ISPASS64960.2025.00015)".
Current: "Its Sapphire Rapids plus H100 launch figure (2.37 microseconds) is a second reference for our P4 check." Replacement: "Its launch figure for a Xeon Platinum 8468V with an H100 PCIe (2.37 microseconds, measured with its own profiler, SKIP) is a second reference for our P4 check."

**B6. Reference 17 (MInference).**
Current: "reports 5% to 30% of time spent building indices on one A100." Replacement: "reports index building rising from 5% to 30% of time as the context falls to 10k tokens, on one A100."

**B7. "How each entry was checked" note.**
Current: "The others are described from their abstracts only."
Replacement: "Entries 17 and 20 had their claims verified against the full text, with machine-checked quotes, on 28 October 2026 (`docs/related_work_verified.md`); this is not a personal reading. The others are described from their abstracts only."

Unchanged after checking: reference 16 (The Sparse Frontier) already uses the paper's narrower wording; reference 18 (XAttention) keeps "ICML", which stays unconfirmed from the proceedings (Section 1, flag 1).

## Open questions for the researcher

1. The brief's note lists entries 11 (CompactAttention), 12 (Vellaisamy et al.) and 16 (The Sparse Frontier) as "read in full". The relayed decision keeps "read in full" for papers the researcher reads personally and names XAttention, KernelSight-LM, TaxBreak and CTFusion. If entries 11, 12 and 16 were not read personally, they should move to "claims verified against full text" as well. Not changed in B7 until confirmed.
2. SpargeAttn joins T7.2, but neither the plan's Section 3 nor the brief's reference list cites it. Whether it gets a reference entry is not covered by the decision.
