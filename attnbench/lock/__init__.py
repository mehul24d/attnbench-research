"""Lock tooling (plan Phase 4, Gate B): the lock-list checker, the fits and
procedures that produce lock values from banked data, and the tag check.

  - `checklist`: the values T4.1 lists, read from the plan, against a lock
    file;
  - `hdbi_fit`: HDBI's one boundedness threshold, fitted by sign agreement
    (Section 5.4);
  - `holm`: the Holm-corrected admissible set (Section 5.7);
  - `floors`: the resolution and noise floors from the measured run-to-run
    spread (Section 5.6).

`scripts/tag_lock.py` checks a lock commit and tags it locally (T4.4).
Nothing here imports torch.
"""
