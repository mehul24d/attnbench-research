# Image `attnbench-env-v6-20260917`: full package list

Recorded on 17 Nov 2026 (the researcher's yes of 17 Nov 2026; `docs/permission_log.md`). **The image is kept**: this list was taken so it can be deleted later, but deleting it needs its own yes. One `e2-micro` VM was booted from the image in `asia-southeast1-b`, with no GPU, no external address and no service account, a 200 GB `pd-standard` boot disk, a self-halt at 20 minutes and a 30-minute run limit that deletes it. A startup script (the same as v5's second pass) wrote the lists to the serial port as a gzipped, base64 tar with its sha256 (`bbd094c2…`). Locally, the tar was reassembled, its sha256 checked, and every file checked against the VM's `SHA256SUMS`.

**Every list here is byte-identical to v5's** (`docs/image_libraries/attnbench-env-v5-20260905/`): `pip freeze` as root, as `ubuntu`, as `mehuldahiya` (162 lines) and of `mehuldahiya`'s user site (56 lines), and `dpkg -l` (687 lines). v6 was captured from a v5 boot (Session 11, 17 Sep 2026, `docs/spend_ledger.md`), and installed nothing at the package level. Only `meta.txt` (the host name and date) differs.

The environment the sessions use is the system Python 3.10.12 (`/usr/bin/python3`) with the system packages and `mehuldahiya`'s user site, `~/.local/lib/python3.10/site-packages`; there is no venv and no conda. Among them: torch 2.9.1+cu129, triton 3.5.1, cuDNN `nvidia-cudnn-cu12==9.10.2.21`, flash_attn 2.8.3.post1, xformers 0.0.33+5d4b92a5.d20251029, flash-linear-attention 0.5.2 and transformers 4.46.0. FA3 is not in the image: the sessions install the pinned wheel at boot (`FA3_WHEEL_SHA256`).

| File | What it is |
|---|---|
| `pip_freeze_user_mehuldahiya.txt` | **The session environment**: `pip freeze --all` as `mehuldahiya`, the system packages and the user site together |
| `pip_freeze_usersite_mehuldahiya.txt` | `pip freeze --user` as `mehuldahiya`: the user site alone |
| `pip_freeze_root.txt`, `pip_freeze_user_ubuntu.txt` | the system packages alone (no user site); identical |
| `pip_list_user_mehuldahiya.json` | `pip list --format=json` as `mehuldahiya` |
| `dpkg_l.txt` | `dpkg -l` (CUDA, cuDNN, the driver packages) |
| `meta.txt` | kernel, users, site directories, the pyvenv.cfg search (none found) |
| `SHA256SUMS` | the VM's checksums; it also lists `pip_list_root.json`, `pip_list_user_ubuntu.json`, the empty `.err` files, `pip_freeze_usersite_ubuntu.txt` (empty) and `usersite_ls_*.txt`, which are kept only in `attnbench_backups/v6_libraries/` |
