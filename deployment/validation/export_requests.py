"""Export recorded observations through the unchanged client encoder; no hardware."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--client-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-tasks", required=True, type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.client_root / "src"))
    import msgpack
    import numpy as np
    from wallx_client.config import PolicyConfig
    from wallx_client.contracts import Observation
    from wallx_client.protocol import encode_request

    seen, cases = set(), []
    for config in sorted(args.rollouts.glob("*/config.json"), reverse=True):
        cfg = json.loads(config.read_text())["configuration"]
        task = cfg["task"]
        queries = sorted(config.parent.glob("query_*.npz"), key=lambda p: p.stat().st_mtime)
        if task in seen or len(queries) < 3:
            continue
        seen.add(task)
        policy = PolicyConfig(**cfg["policy"])
        for index in [0, len(queries) // 2, len(queries) - 1]:
            source = queries[index]
            with np.load(source, allow_pickle=False) as data:
                obs = Observation(
                    state=data["state"],
                    views={name: data[name] for name in policy.camera_names},
                    captured_monotonic=0,
                    captured_wall=0,
                    state_monotonic=0,
                )
                _, wire = encode_request(obs, task, policy)
            body = msgpack.unpackb(wire, raw=False)
            body["request_id"] = f"parity-{len(cases):02d}"
            cases.append(
                {
                    "id": body["request_id"],
                    "task": task,
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "request": body,
                }
            )
    if args.expected_tasks < 1 or len(seen) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} tasks, got {len(seen)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {"cases": len(cases), "tasks": sorted(seen), "bytes": args.output.stat().st_size}
        )
    )


if __name__ == "__main__":
    main()
