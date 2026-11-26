# Launch card: P6 (AMD EPYC 7R13 + H100, AWS `p5.4xlarge`, Sydney; EXPLORATORY)

Prepared 26 Nov 2026 (the researcher's post-run task of 26 Nov 2026, item 6). **Not built and
not launched; nothing here launches anything.** A launch needs the items under "Owed before a
launch yes" and then the researcher's yes. P6 is exploratory (Appendix A, 21 Nov 2026): it
scores no hypothesis.

## The gate: run

The host-effect rule (Appendix A, 21 Nov 2026; `scripts/p6_host_effect.py`) on P3 against P1,
read after P3's block 2 on 26 Nov 2026 (record `attnbench_backups/aws_p3_g6_20261126T0906Z/`,
`gate.json`): all 35 cells, **d = 0.2440** (p95 0.2699, total ratio − 1 = −0.2389).
d > 0.10, so **P6 runs**. P3's host components are about 24% cheaper in total than P1's.

## The session

| Item | Value |
|---|---|
| Instance | `p5.4xlarge`: 1x H100 80GB, 16 vCPU, AMD EPYC 7R13 (`scripts/aws_launch_session.sh P6`) |
| Region | `ap-southeast-2` (Sydney) only, on demand (rule 12, Appendix A, 21 Nov 2026) |
| Zones | not read yet: `describe-instance-type-offerings` for `p5.4xlarge` in Sydney, before the yes |
| Quota | Sydney's P-instance quota is 16 vCPU: one `p5.4xlarge` exactly, nothing beside it |
| Image | the environment build's `ami-096c5dc30364c9041` (22 Nov 2026) |
| Cells | P4's 35, `docs/p6_cells.json` (sha256 `a881bce6…`), cell for cell against P4 |
| Steps | P4's 16, one session (no Spot blocks) |
| Launch deadline | **2 Dec 2026, 18:30 UTC** (epoch 1796236200) |

## Hours, rate, cap

Planned **4.0 h** (Appendix A, 21 Nov 2026): setup on the built image, preflight and
bootstrap check 0.50; the suite record 0.33; the steps 1.90 (P4 measured 74 min in all;
the host-bound steps doubled for the AMD host, 114 min); teardown, pull and cleanup check
0.15; margin 1.12. **Plan default, flagged:** as for P3, the suite record is made on the Mac
after the run, so its 0.33 h becomes margin.

Rate **₹990.2/h** all-in, the card rate of Appendix A, 21 Nov 2026: (the instance at
$8.944/h, Sydney's public price file of 20 Nov 2026, + 200 GB gp3 at $0.096/GB-month over
730 h) × 1.15 × ₹95.99. Not re-read for this card (no AWS call was made); re-read before the
yes, and the card re-priced if it moved.

| | Minutes | ₹ at the rate |
|---|---|---|
| Planned | 240 | 3,961 |
| In-guest halt | **312** | 5,149 |
| Hard delete | **359** | 5,925 (worst case) |
| **Cap** | | **5,940** (4.0 h × ₹990.2 × 1.5) |

## Budget

AWS cap about ₹10,400 (Appendix A, 17 Nov 2026). Spent on AWS to 26 Nov 2026 at the all-in
rate: about ₹963 (to 22 Nov ₹204; the 23 Nov rehearsal ₹36; the 25 Nov rehearsal ₹41; P3's
aborted session ₹143; P3's run ₹539), plus the image's snapshot at about ₹442 a month. With
P6's cap: about ₹6,903 plus the snapshot, inside the cap. Credits: $159.79 read on 23 Nov 2026
(about ₹15,338), not re-read since.

## Owed before a launch yes

1. **P6's committed predictions** (a frozen-pipeline and an FA3-backend-corrected variant):
   they need their own ruling and must be committed before P6's ground truth (Appendix A,
   21 Nov 2026). Their dense cells should name the kernel ground truth will run (P3 and P4
   both priced `sdpa_flash` while ground truth ran the dense selection's choice).
2. **A P6 driver.** `scripts/aws_p3_driver.sh` is P3's (its three blocks, its gate). P6 needs
   P4's 16 steps on one instance, with P3's launch path, halts and teardown, including this
   wave's halt push at each pull and the optional final re-pull.
3. **The image on an H100.** The image was built and rehearsed on the L4 (sm_89); its FA3
   and block-sparse builds must be checked on sm_90 before any timed step.
4. **Zones and capacity** for `p5.4xlarge` in Sydney, and the quota, read back.
5. **The login.** `aws login` sessions end at most 12 h after the login, so the window and the
   run need a fresh login, and the security group has to admit the Mac's address at launch
   (it changed twice during P3).
