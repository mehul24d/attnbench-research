"""scripts/prompt_set_hash.py: the hash and the row check, on fixed entries
(the build itself needs the pinned RULER data and tokenizer)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prompt_set_hash as H  # noqa: E402


def test_the_hash_moves_with_any_token_and_not_with_key_order():
    a = [H.entry("qa_1", 2048, 0, "qa_1_2048_0", [1, 2, 3]),
         H.entry("qa_1", 2048, 1, "qa_1_2048_1", [1, 2, 4])]
    assert a[0]["n_tokens"] == 3 and a[0]["ids_sha256"] != a[1]["ids_sha256"]
    h = H.set_sha256(a)
    assert H.set_sha256([dict(reversed(list(e.items()))) for e in a]) == h
    assert H.set_sha256([a[0], H.entry("qa_1", 2048, 1, "qa_1_2048_1", [1, 2, 5])]) != h
    assert H.set_sha256(a[::-1]) != h                      # the order is part of the set
    assert H.entry("qa_1", 2048, 0, "x", [12, 3])["ids_sha256"] != \
        H.entry("qa_1", 2048, 0, "x", [1, 23])["ids_sha256"]


def test_the_row_check_names_each_length_whose_recorded_tokens_differ(tmp_path):
    es = [H.entry("qa_1", 2048, 0, "a", [1] * 5), H.entry("qa_1", 2048, 1, "b", [1] * 9),
          H.entry("qa_1", 4096, 0, "c", [1] * 7), H.entry("qa_1", 8192, 0, "d", [1] * 8)]
    (tmp_path / "gt_L2048_b1.json").write_text(json.dumps({"rows": [{"context_length": 5}]}))
    (tmp_path / "gt_L4096_b1.json").write_text(json.dumps({"rows": [{"context_length": 6}]}))
    assert H.check_rows(es, tmp_path) == [(4096, 7, [6]), (8192, 8, None)]
