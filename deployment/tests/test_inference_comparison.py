"""The offline gate must reject misleading matches, not merely parse artifacts."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "corruption", [None, "preprocessing", "action", "coverage", "decoder", "repeat"]
)
def test_parity_gate_rejects_invalid_evidence(tmp_path, corruption):
    row = {
        "case": "test",
        "seed": 7,
        "repeat": 0,
        "raw_actions": [[1.0] * 26 for _ in range(32)],
        "client_actions": [[1.0] * 7 for _ in range(32)],
        "prepared": {"input_ids": {"sha256": "original"}},
    }
    repeat = copy.deepcopy(row)
    repeat["repeat"] = 1
    baseline = {
        "metadata": {},
        "requests_sha256": "same",
        "checkpoint_assets": {},
        "rows": [row, repeat],
    }
    candidate = copy.deepcopy(baseline)
    if corruption == "preprocessing":
        candidate["rows"][0]["prepared"]["input_ids"]["sha256"] = "different"
    elif corruption in ("action", "repeat"):
        candidate["rows"][0]["raw_actions"][0][0] = 2.0
        candidate["rows"][0]["client_actions"][0][0] = 2.0
        if corruption == "repeat":
            baseline = copy.deepcopy(candidate)
    elif corruption == "coverage":
        candidate["rows"].pop()
        baseline["rows"].pop()
    elif corruption == "decoder":
        candidate["rows"][0]["client_actions"][0][0] = 2.0
    for name, value in [("baseline", baseline), ("candidate", candidate)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    script = Path(__file__).resolve().parents[1] / "validation/compare_inference.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(tmp_path / "baseline.json"),
            str(tmp_path / "candidate.json"),
            "--output",
            str(tmp_path / "result.json"),
            "--expected-cases",
            "1",
            "--seeds",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    evidence = json.loads((tmp_path / "result.json").read_text())
    assert result.returncode == (0 if corruption is None else 1), result.stderr
    assert evidence["passed"] is (corruption is None)
