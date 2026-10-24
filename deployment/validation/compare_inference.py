"""Compare replay artifacts with a fixed, strict numerical gate."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=9)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 41])
    args = parser.parse_args()
    old, new = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
    failures, cases = [], []
    for key in ("requests_sha256", "checkpoint_assets", "metadata"):
        if old[key] != new[key]:
            failures.append(f"{key} differs")

    def index(report):
        return {(r["case"], r["seed"], r["repeat"]): r for r in report["rows"]}

    a, b = index(old), index(new)
    if a.keys() != b.keys() or not a:
        failures.append("Replay case sets differ or are empty")
    for label, data in [("baseline", a), ("candidate", b)]:
        ids = {key[0] for key in data}
        expected = {
            (case, seed, repeat) for case in ids for seed in args.seeds for repeat in (0, 1)
        }
        if len(ids) != args.expected_cases or data.keys() != expected:
            failures.append(f"{label}: incomplete observation/seed/repeat coverage")
    for key in sorted(a.keys() & b.keys()):
        if a[key]["prepared"] != b[key]["prepared"]:
            failures.append(f"{key}: preprocessed model inputs differ")
        x, y = [np.asarray(r[key]["raw_actions"], dtype=np.float32) for r in (a, b)]
        finite = np.isfinite(x).all() and np.isfinite(y).all()
        passed = (
            x.shape == y.shape
            and x.shape in ((32, 26), (1, 32, 26))
            and finite
            and np.allclose(x, y, rtol=1e-5, atol=1e-5)
        )
        if not passed:
            failures.append(f"{key}: raw action comparison failed")
        cases.append(
            {
                "key": key,
                "max_abs_error": float(np.max(np.abs(x - y)))
                if x.shape == y.shape and finite
                else None,
                "bitwise_equal": np.array_equal(x, y),
                "passed": bool(passed),
            }
        )
        for label, data in [("baseline", a), ("candidate", b)]:
            raw = np.asarray(data[key]["raw_actions"], dtype=np.float32)
            client = np.asarray(data[key]["client_actions"], dtype=np.float32)
            if (
                raw.shape not in ((32, 26), (1, 32, 26))
                or client.shape != (32, 7)
                or not np.array_equal(client, raw.reshape(32, 26)[:, :7])
            ):
                failures.append(f"{label} {key}: client action decoding differs from raw response")
            if key[2] == 1:
                if (key[0], key[1], 0) not in data:
                    continue
                first = np.asarray(data[(key[0], key[1], 0)]["raw_actions"])
                second = np.asarray(data[key]["raw_actions"])
                if not np.array_equal(first, second):
                    failures.append(f"{label} {key}: seeded repeat was not bitwise reproducible")
    for label, report in [("baseline", old), ("candidate", new)]:
        if len(report["rows"]) != len(index(report)):
            failures.append(f"{label}: duplicate case keys")
    result = {
        "passed": not failures,
        "rtol": 1e-5,
        "atol": 1e-5,
        "cases": cases,
        "failures": failures,
        "scope": "Real weights, recorded observations, shared client decoder; no physical robot validation",
        "backends": {
            "baseline": old.get("backend", "websocket"),
            "candidate": new.get("backend", "websocket"),
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
