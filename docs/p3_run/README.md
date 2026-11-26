# P3's run, 26 Nov 2026 (EXPLORATORY)

The relaunch (the researcher's yes of 26 Nov 2026), commit `6e9188a`, one `g6.2xlarge` in
`ap-southeast-2a`, launched 09:12 UTC, terminated 12:57:34 UTC, about ₹539 (226 min at
₹143.21/h). The full record, the session tree with all 12 steps, stays off-repo in
`attnbench_backups/aws_p3_g6_20261126T0906Z/` (about 8 GB).

| File | What |
|---|---|
| `pulls.tsv` | each block's pull: files, and the tar's sha256 (every file's sha256 checked on the Mac) |
| `gate.json`, `gate.txt` | the host-effect rule after block 2: 35 cells, d = 0.2440, verdict run |
| `p3_exploratory_scored.json`, `.md` | the 22 Nov 2026 prediction against P3's ground truth, EXPLORATORY (`scripts/score_p3_exploratory.py`) |
| `suite_record.md`, `prelaunch_cpu_pass_p3_6e9188a.json` | the suite record, CPU only, at `6e9188a` |

## Blocks

| Block | Ran (UTC) | Steps | Pull |
|---|---|---|---|
| b1 | 09:18 to 09:38 | preflight, link, dense_select, probe | 274 files |
| b2 | 09:52 to 10:58 | host_components | 693 files |
| b3 | 11:13 to 12:28 | device_components to delay_injection | 349 files, 2.68 GB |

SSH was lost from 11:59 to 12:09 UTC in block 3, when this Mac's address changed; the
researcher re-pointed the security group (`sg_repoint.sh`) and the driver went on.

## The final re-pull that failed

After block 3's pull ended (12:52:21 UTC), the driver re-pulled every step directory as one
tar, about 8 GB; its pack on the instance succeeded, and its `scp` failed at 12:56:32 UTC,
then the teardown's SSH reads failed too, with the instance running. **Likely cause:**
block 3's idle halt was due at 12:58:10 UTC, and `shutdown` writes `/run/nologin` 5 minutes
before a scheduled halt (12:53:10 UTC); the pack ran on a login opened before then, and the
`scp`, a new login opened after the 4-minute pack, was refused. Not proven: ssh's error
output was discarded. Nothing was lost (blocks 1 to 3 had been pulled and checked). Since
26 Nov 2026 the re-pull is optional (`P3_DRIVER_FINAL_REPULL=1`), each pull pushes the halt
back (`aws_p3_instance.sh push-halt`), and scp's error is kept, masked.

## Identifiers

The cleanup check's listings in the regions the organization's policy denies printed the
policy's ARN, the role and its session id into every driver log since 21 Nov 2026 (the
account number was masked). They were redacted in place on 26 Nov 2026 (18 files; the
manifest of hashes before and after is `attnbench_backups/REDACTIONS_20261126.tsv`), and the
check now masks those errors at the source.
