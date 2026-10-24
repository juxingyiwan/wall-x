"""Check maintained documentation links and examples without model/device access.

Run from any directory: python /path/to/repository/scripts/check_docs.py
Requires the base client dependencies. External URLs and hardware instructions
are intentionally not executed.
"""

import argparse
import ast
import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(text):
    """Omit fenced code so example paths are not mistaken for Markdown links."""
    marker = None
    for line in text.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            token = fence.group(1)
            if marker is None:
                marker = token
            elif token[0] == marker[0] and len(token) >= len(marker):
                marker = None
            continue
        if marker is None:
            yield line


def anchors(path):
    found, counts = set(), {}
    for line in prose(path.read_text(encoding="utf-8")):
        found.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', line))
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            value = re.sub(r"<[^>]+>", "", heading.group(1)).lower()
            value = re.sub(r"[^\w\- ]", "", value).replace(" ", "-")
            n = counts.get(value, 0)
            counts[value] = n + 1
            found.add(value if n == 0 else f"{value}-{n}")
    return found


def maintained_docs(*, include_local=False):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
    )
    if result.returncode == 0:
        names = result.stdout.decode().split("\0")
    else:
        names = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.md")]
    if include_local:
        names.extend(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "deployment/local").rglob("*.md")
        )
    return sorted(
        {
            ROOT / n
            for n in names
            if n.endswith(".md")
            and (ROOT / n).is_file()
            and (include_local or not n.startswith("deployment/local/"))
            and (
                n.startswith(("docs/", "deployment/", "environments/", "workspace/"))
                or n
                in {
                    "README.md",
                    "README.zh-CN.md",
                    "README.upstream.md",
                    "README.upstream.zh-CN.md",
                    "CONTRIBUTING.zh-CN.md",
                    "CONTRIBUTING.md",
                    "scripts/README.md",
                    "tools/README.md",
                    "workspace/README.md",
                    "THIRD_PARTY_NOTICES.md",
                }
            )
        }
    )


def relative_links(path):
    """Yield local links, including their original spelling for diagnostics."""
    for line in prose(path.read_text(encoding="utf-8")):
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", line):
            target = target.split(' "', 1)[0].strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            dest = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            yield target, dest, unquote(parts.fragment)


def check_links(docs=None):
    count, errors = 0, []
    docs = maintained_docs() if docs is None else docs
    for path in docs:
        for target, dest, fragment in relative_links(path):
            count += 1
            if not dest.is_relative_to(ROOT) or not dest.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing target {target}")
            elif fragment and dest.suffix == ".md" and fragment not in anchors(dest):
                errors.append(f"{path.relative_to(ROOT)}: missing anchor {target}")
    if errors:
        raise ValueError("\n".join(errors))
    return len(docs), count


def check_navigation(docs=None):
    """Every maintained public page must be reachable from the main index."""
    public = set(maintained_docs() if docs is None else docs)
    pending = [ROOT / "README.md"]
    reached = set()
    while pending:
        path = pending.pop()
        if path in reached or path not in public:
            continue
        reached.add(path)
        pending.extend(dest for _, dest, _ in relative_links(path) if dest in public)
    missing = sorted(str(p.relative_to(ROOT)) for p in public - reached)
    if missing:
        raise ValueError(
            "Pages missing navigation from README.md: " + ", ".join(missing)
        )
    return len(reached)


def check_homing_constants():
    """Read source literals with AST; never import the hardware adapter."""
    page = ROOT / "README.md"
    documented = {
        (module, name): json.loads(value)
        for module, name, value in re.findall(
            r"^\| `(front_home|zero_servo)\.([A-Z][A-Z0-9_]*)` \| `(.*?)` \|",
            page.read_text(encoding="utf-8"),
            re.M,
        )
    }
    if not documented:
        raise ValueError("README.md is missing the homing constants table")
    for module in {key[0] for key in documented}:
        path = ROOT / f"deployment/client/src/wallx_client/hardware/{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = node.value.value
        for (owner, name), value in documented.items():
            if owner == module and (name not in values or values[name] != value):
                raise ValueError(f"README.md: {module}.{name} differs from source")
    return len(documented)


def wallx_commands(text):
    """Extract literal public CLI examples; do not evaluate shell code."""
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line, comments=True)
            if words and words[0].startswith("WALLX_PADDING="):
                words = words[1:]
            if words[:2] == ["bash", "wallx"]:
                if any("$(" in word or "`" in word for word in words):
                    raise ValueError(
                        "Documented wallx commands must use literal arguments"
                    )
                yield words[2:]


def check_command_examples():
    """Use the actual parsers, substituting fixtures and stopping before I/O."""
    imported_before = set(sys.modules)
    # No public check depends on ignored profiles, real devices or model assets.
    sys.path[:0] = [str(ROOT), str(ROOT / "deployment/client/src")]
    import yaml
    from wallx_client.cli import apply_cli_overrides, build_parser
    from wallx_client.config import load_config

    from deployment import cli, local_runtime

    # The fixture describes a hypothetical GPU host; it must also parse on CPU
    # CI hosts with an empty CUDA_VISIBLE_DEVICES or inherited Slurm variables.
    # These values are restored before command inspection and never reach a GPU.
    with patch.dict(os.environ, {"WALLX_PADDING": "zero", "CUDA_VISIBLE_DEVICES": "0"}):
        os.environ.pop("SLURM_JOB_ID", None)
        fixture = cli.load_deployment(
            ROOT / "deployment/configs/b601.example.yml",
            ROOT / "deployment/machines/standalone.example.yml",
        )
    pages = [
        ROOT / "docs/rebot.md",
        ROOT / "docs/rebot.zh-CN.md",
    ]

    def inspect_client(config, argv, **_kwargs):
        with tempfile.TemporaryDirectory(prefix="wallx-docs-") as directory:
            path = Path(directory) / "client.yml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            parsed = build_parser().parse_args(["--config", str(path), *argv])
            apply_cli_overrides(load_config(path, validate=False), parsed)
        return 0

    def inspect_local(deployment, args):
        argv = list(args.client_args)
        if argv[:1] == ["--"]:
            argv.pop(0)
        if not argv or argv == ["check"]:
            return 0
        return inspect_client(
            local_runtime.local_deployment(deployment)["client"], argv
        )

    def inspect_server(deployment):
        # Validate command construction, then prevent dispatch by raising a
        # private sentinel. A missing --dry-run in prose cannot launch a server.
        cli_server_command(deployment)
        raise ServerInspected

    class ServerInspected(Exception):
        pass

    def forbid_exec(*_args, **_kwargs):
        raise AssertionError("Documentation validation attempted to launch a process")

    cli_server_command = cli.server_command
    count = 0
    with (
        patch.object(
            cli,
            "load_deployment",
            side_effect=lambda *_a, **_kw: copy.deepcopy(fixture),
        ),
        patch.object(cli, "run_client", side_effect=inspect_client),
        patch.object(local_runtime, "run_local", side_effect=inspect_local),
        patch.object(cli, "server_command", side_effect=inspect_server),
        patch.object(cli.os, "execvpe", side_effect=forbid_exec),
    ):
        for page in pages:
            for argv in wallx_commands(page.read_text(encoding="utf-8")):
                try:
                    # Capture help/config output; only the compact check summary
                    # belongs in CI logs. None of these branches opens hardware.
                    with contextlib.redirect_stdout(io.StringIO()):
                        try:
                            result = cli.main(argv)
                        except ServerInspected:
                            result = 0
                        except SystemExit as exc:
                            result = exc.code
                    if result != 0:
                        raise ValueError(f"parser returned {result}")
                except (ValueError, AssertionError) as exc:
                    raise ValueError(
                        f"{page.relative_to(ROOT)}: bash wallx {shlex.join(argv)}: {exc}"
                    ) from exc
                count += 1
    if count == 0:
        raise ValueError("No public wallx command examples found")
    newly_imported = sys.modules.keys() - imported_before
    if {"torch", "motorbridge", "can", "lerobot_robot_seeed_b601"} & newly_imported:
        raise RuntimeError("Command validation imported model/hardware modules")
    return count


def check_examples():
    # Base client imports are deliberately free of Torch and hardware drivers.
    sys.path.insert(0, str(ROOT / "deployment/client/src"))
    import yaml
    from wallx_client import config
    from wallx_client.protocol import validate_metadata

    manifest = json.loads(
        (ROOT / "deployment/client/docs/examples/server_manifest.json").read_text()
    )
    profile = yaml.safe_load((ROOT / "deployment/configs/b601.example.yml").read_text())
    policy = config.PolicyConfig(
        checkpoint_id=profile["model"]["checkpoint_id"],
        checkpoint_sha256=profile["model"]["checkpoint_sha256"],
        padding=profile["padding"],
    )
    validate_metadata(manifest, policy)
    table = (ROOT / "docs/rebot.md").read_text(encoding="utf-8")
    documented = {
        name: json.loads(value)
        for name, value in re.findall(r"^\| `([\w.]+)` \| `(.*?)` \|$", table, re.M)
    }
    actual = {}
    for prefix, cls in (
        ("policy", config.PolicyConfig),
        ("robot", config.RobotConfig),
        ("control", config.ControlConfig),
        ("rtc", config.RTCConfig),
        ("shutdown", config.ShutdownConfig),
    ):
        for key, value in dataclasses.asdict(cls()).items():
            actual[f"{prefix}.{key}"] = json.loads(json.dumps(value))
    for field in dataclasses.fields(config.Config):
        if field.default is not dataclasses.MISSING:
            actual[field.name] = field.default
    required = {
        "policy.horizon",
        "policy.action_dims",
        "policy.padding",
        "control.hz",
        "control.execution_horizon",
        "control.duration_s",
        "control.startup_home",
        "shutdown.home_timeout_s",
        "rtc.enabled",
        "rtc.guidance_mode",
        "display_data",
        "save_rollout",
        "display_port",
    }
    if set(documented) != required or any(
        actual[k] != v for k, v in documented.items()
    ):
        raise ValueError("Client default reference differs from config.py")
    template = ROOT / "workspace/example/b601.yml"
    legacy_template = ROOT / "docs/b601/examples/train.example.yml"
    if template.read_bytes() != legacy_template.read_bytes():
        raise ValueError(
            "B601 compatibility template differs from workspace/example/b601.yml"
        )
    training = yaml.safe_load(template.read_text())
    for key in ("dof_config", "ar_dof_config", "agent_pos_config"):
        if list(training["task"][key].values()) != [6, 1, 19]:
            raise ValueError(f"Public training example has an invalid {key}")
    if training["data"]["num_workers"] != 0:
        raise ValueError("Public resume example requires num_workers=0")
    if (
        training["data"]["lerobot_config"]["b601"]["statistics_scope"]
        != "train_episodes"
    ):
        raise ValueError("Public data tutorial expects training-episode statistics")
    if {"torch", "motorbridge", "can", "lerobot_robot_seeed_b601"} & sys.modules.keys():
        raise RuntimeError("Documentation validation imported model/hardware modules")
    return len(documented)


PAIRS = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/libero.md",
    "docs/robotwin.md",
    "docs/rebot.md",
)


def check_bilingual(pairs=PAIRS):
    """Reject drifting executable examples, anchors and literal parameters."""
    for name in pairs:
        english = (ROOT / name).read_text(encoding="utf-8")
        chinese = (ROOT / name.replace(".md", ".zh-CN.md")).read_text(encoding="utf-8")
        for label, pattern in (
            ("code blocks", r"```[^\n]*\n.*?```"),
            ("section anchors", r'<a id="[^"]+">'),
        ):
            if re.findall(pattern, english, re.S) != re.findall(pattern, chinese, re.S):
                raise ValueError(f"{name}: bilingual {label} differ")
        if Counter(re.findall(r"(?<!`)`([^`\n]+)`(?!`)", english)) != Counter(
            re.findall(r"(?<!`)`([^`\n]+)`(?!`)", chinese)
        ):
            raise ValueError(f"{name}: bilingual literal parameters differ")
        if len(re.findall(r"^#{1,6} ", english, re.M)) != len(
            re.findall(r"^#{1,6} ", chinese, re.M)
        ):
            raise ValueError(f"{name}: bilingual section counts differ")
    return len(pairs)


def check_upstream():
    """Verify retained originals against a manifest exported from official Git."""
    manifest = json.loads(
        (ROOT / "tests/fixtures/upstream_documentation.json").read_text()
    )
    for name, entry in manifest["documents"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(
                f"{name}: differs from official {manifest['upstream_commit']}"
            )
    expected = set(manifest["documents"]) - {".github/ISSUE_TEMPLATE/bug_report.md"}
    expected |= set(PAIRS) | {n.replace(".md", ".zh-CN.md") for n in PAIRS}
    actual = {p.relative_to(ROOT).as_posix() for p in maintained_docs()}
    if actual != expected:
        raise ValueError(f"Documentation catalog differs: {sorted(actual ^ expected)}")
    return len(manifest["documents"])


def check_workflow_examples():
    """Check shell syntax and documented training flags without launching them."""
    entries = {
        "wall_x.trainer.fsdp_trainer.train_fsdp": "wall_x/trainer/fsdp_trainer/train_fsdp.py",
        "scripts/compute_norm_stats.py": "scripts/compute_norm_stats.py",
        "scripts.prepare_b601": "scripts/prepare_b601.py",
        "scripts.validate_b601_data": "scripts/validate_b601_data.py",
        "scripts.train_b601": "scripts/train_b601.py",
    }
    options = {}
    for entry, source in entries.items():
        tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
        options[entry] = {"--help", "-h"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_argument":
                    options[entry].update(
                        arg.value
                        for arg in node.args
                        if isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith("-")
                    )
    blocks = commands = 0
    for name in PAIRS:
        for page in (name, name.replace(".md", ".zh-CN.md")):
            text = (ROOT / page).read_text(encoding="utf-8")
            for block in re.findall(r"```bash\n(.*?)```", text, re.S):
                result = subprocess.run(
                    ["bash", "-n"], input=block, text=True, capture_output=True
                )
                if result.returncode:
                    raise ValueError(f"{page}: invalid shell example: {result.stderr}")
                blocks += 1
                for line in block.replace("\\\n", " ").splitlines():
                    tokens = shlex.split(line, comments=True)
                    if tokens[:2] == ["huggingface-cli", "download"]:
                        raise ValueError(f"{page}: use hf download")
                    if tokens[:2] == ["hf", "download"] and "--revision" not in tokens:
                        raise ValueError(
                            f"{page}: pin downloaded assets with --revision"
                        )
                    for entry, accepted in options.items():
                        if entry not in tokens:
                            continue
                        if entry.endswith("train_fsdp") and tokens[0] != "torchrun":
                            raise ValueError(
                                f"{page}: native training requires torchrun"
                            )
                        for token in tokens[tokens.index(entry) + 1 :]:
                            flag = token.split("=", 1)[0]
                            if flag.startswith("--") and flag not in accepted:
                                raise ValueError(
                                    f"{page}: {entry} has no option {flag}"
                                )
                        commands += 1
    return {"bash_blocks": blocks, "training_data_commands": commands}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also check links in existing private site Markdown",
    )
    args = parser.parse_args(argv)
    documents, links = check_links(maintained_docs(include_local=args.include_local))
    reachable = check_navigation()
    bilingual = check_bilingual()
    upstream = check_upstream()
    workflows = check_workflow_examples()
    defaults = check_examples()
    homing_constants = check_homing_constants()
    commands = check_command_examples()
    print(
        json.dumps(
            {
                "documents": documents,
                "bilingual_pairs": bilingual,
                "unmodified_upstream_files": upstream,
                "workflow_examples": workflows,
                "relative_links": links,
                "client_defaults": defaults,
                "reachable_public_documents": reachable,
                "homing_constants": homing_constants,
                "command_examples": commands,
                "manifest_example": "valid",
                "training_example": "valid",
                "hardware_opened": False,
            }
        )
    )


if __name__ == "__main__":
    main()
