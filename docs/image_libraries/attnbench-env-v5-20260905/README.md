# Image `attnbench-env-v5-20260905`: full package list

Recorded on 16 Nov 2026, before the image was deleted (the researcher's yes of 16 Nov 2026; `docs/permission_log.md`). One `e2-micro` VM was booted from the image, with no GPU, no external address and no service account. A startup script wrote the lists to the serial port as a gzipped, base64 tar with its sha256. Locally, the tar was reassembled, its sha256 checked, and every file checked against the VM's `SHA256SUMS`. The VM ran from 17:58:10Z to its deletion at 18:05:49Z.

The environment the sessions used is the system Python 3.10.12 (`/usr/bin/python3`). The system packages sit in `/usr/local/lib/python3.10/dist-packages` and `/usr/lib/python3/dist-packages`; the user `mehuldahiya` has a user site, `~/.local/lib/python3.10/site-packages`. There is no venv and no conda.

| File | What it is |
|---|---|
| `pip_freeze_user_mehuldahiya.txt` | **The session environment**: `pip freeze --all` as `mehuldahiya`, the system packages and the user site together, 162 lines |
| `pip_freeze_usersite_mehuldahiya.txt` | `pip freeze --user` as `mehuldahiya`: the user site alone, 56 lines (flash_attn, xformers, flash-linear-attention and the editable `attnbench`) |
| `pip_freeze_root.txt`, `pip_freeze_user_ubuntu.txt` | the system packages alone (no user site), 110 lines each |
| `pip_list_user_mehuldahiya.json` | `pip list --format=json` as `mehuldahiya` |
| `dpkg_l.txt` | `dpkg -l`, 687 lines (CUDA, cuDNN, the driver packages) |
| `meta.txt` | kernel, users, site directories, the pyvenv.cfg search (none found) |
| `SHA256SUMS` | the VM's checksums; it also lists `pip_list_root.json`, `pip_list_user_ubuntu.json`, the `.err` files and `usersite_ls_*.txt`, which are kept only in `attnbench_backups/v5_libraries/` |

The list agrees with the 8 libraries the v5 sessions recorded in `versions.json` (H100 stage 1, September 2026):
- Python 3.10.12;
- torch 2.9.1+cu129;
- triton 3.5.1;
- cuDNN 91002 (`nvidia-cudnn-cu12==9.10.2.21`);
- flash_attn 2.8.3.post1;
- xformers 0.0.33+5d4b92a5.d20251029;
- flash-linear-attention 0.5.2.

The first pass (root only) missed the user site; its files are kept in the backups and not committed here.
