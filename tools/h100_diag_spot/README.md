# The H100 diagnostic's Spot driver (EXPLORATORY)

`diag_spot.sh` is research-79's guarded copy of the diagnostic's Spot driver (17 Nov 2026): one lock for every copy (`~/.attnbench/h100_diag.lock`), refused if any `attnbench-h100-diag-*` VM is listed, `--window-override` for the first session (the researcher's yes of 17 Nov 2026) and `--rerun` for the two-cell rerun (`docs/exploratory/h100_diag_rerun_card.md`). Its header lists every check. `spot_chain.sh` is P4's setup chain, unchanged; `preempt_cost.py` books a reclaim from Compute Engine's operation times.

The mock harness runs the whole path with `gcloud`, the launcher, the clock and the instance mocked. It uses BSD `date -j` and `sed -i ''`, so it runs on the Mac and is not part of the pytest suite:

    bash tools/h100_diag_spot/test/run_mock.sh rerun_clean        # one scenario; see its case list
    RERUN=1 bash tools/h100_diag_spot/test/run_concurrent.sh       # two starts on one lock: one request

Run the driver from a scratch copy, never from this checkout (the deploy bundles `$REPO`'s HEAD, and a running bash script must not change under it), with `DIAG_DIR` outside the repository.
