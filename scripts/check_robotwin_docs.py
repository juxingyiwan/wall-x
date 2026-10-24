#!/usr/bin/env python3
"""Check maintained RoboTwin docs and examples without model/simulator imports."""

import ast
import re
import shlex
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "README.zh-CN.md", "docs/robotwin.md", "docs/robotwin.zh-CN.md"]
ENTRY_MODULES = {
    "scripts/train_robotwin.py": "wall_x/robotwin/train.py",
    "scripts/infer_robotwin.py": "wall_x/robotwin/inproc.py",
    "scripts/run_robotwin.sh": "wall_x/robotwin/inproc.py",
    "scripts/convert_robotwin.py": "wall_x/robotwin/convert.py",
    "scripts/compute_robotwin_norm_stats.py": "wall_x/robotwin/norm_stats.py",
    "wall_x.robotwin.preflight": "wall_x/robotwin/preflight.py",
    "wall_x.robotwin.convert_lerobot": "wall_x/robotwin/convert_lerobot.py",
}


def anchors(text):
    result = set(re.findall(r'<a\s+id=["\']([^"\']+)', text))
    used = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if match:
            slug = "".join(
                c
                for c in match[1].lower()
                if c in "-_ " or unicodedata.category(c)[0] in "LN"
            ).replace(" ", "-")
            count = used.get(slug, 0)
            used[slug] = count + 1
            result.add(slug + (f"-{count}" if count else ""))
    return result


def cli_options(module):
    """Read argparse declarations; do not import GPU or simulator dependencies."""
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    options = {"--help", "-h"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("-"):
                        options.add(arg.value)
    return options


def check_documents(errors):
    links = commands = 0
    options = {entry: cli_options(module) for entry, module in ENTRY_MODULES.items()}
    for name in DOCS:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
        for match in re.finditer(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", prose):
            target = match[1]
            if re.match(r"\w+://|mailto:", target):
                continue
            links += 1
            file, _, anchor = target.partition("#")
            destination = path.parent / unquote(file) if file else path
            if not destination.exists():
                errors.append(f"{name}: missing link {target}")
            elif anchor and (
                not destination.is_file()
                or unquote(anchor)
                not in anchors(destination.read_text(encoding="utf-8"))
            ):
                errors.append(f"{name}: missing anchor {target}")
        for block in re.finditer(r"^```(?:bash|sh)\s*\n(.*?)^```", text, re.M | re.S):
            lines = block[1].replace("\\\n", " ").splitlines()
            for line in lines:
                if not any(entry in line for entry in options):
                    continue
                tokens = shlex.split(line, comments=True)
                for entry, accepted in options.items():
                    if entry not in tokens:
                        continue
                    commands += 1
                    for token in tokens[tokens.index(entry) + 1 :]:
                        flag = token.split("=", 1)[0]
                        if flag.startswith("--") and flag not in accepted:
                            errors.append(f"{name}: {entry} has no option {flag}")
    return links, commands


def check_templates(errors):
    configs = {}
    for path in sorted((ROOT / "workspace/example").glob("robotwin*.yml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        configs[path.name] = cfg
        checkpoint = cfg.get("checkpoint", {})
        dataset = cfg.get("data", {}).get("lerobot_config", {})
        key = dataset.get("repo_id", "")
        problems = []
        if not str(checkpoint.get("resume_from", "")).endswith(".safetensors"):
            problems.append("new-run example must set resume_from to model weights")
        if checkpoint.get("resume_model"):
            problems.append("resume_model is ignored by the trainer")
        if not key or any(c in key for c in ".\\/") or not dataset.get("root"):
            problems.append("use a logical repo_id and separate root path")
        if dataset.get("adapter") != "robotwin":
            problems.append("missing RoboTwin adapter")
        if not cfg.get("data", {}).get("episode_split_path"):
            problems.append("missing episode split")
        for problem in problems:
            errors.append(f"{path.relative_to(ROOT)}: {problem}")
    canonical = configs.get("robotwin_full_v1.yml")
    if canonical is None or canonical != configs.get("robotwin_full.template.yml"):
        errors.append("Full compatibility template differs from robotwin_full_v1.yml")
    return len(configs)


def main():
    errors = []
    links, commands = check_documents(errors)
    templates = check_templates(errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"RoboTwin docs OK: {len(DOCS)} pages, {links} links/anchors, "
        f"{commands} CLI examples, {templates} templates"
    )


if __name__ == "__main__":
    main()
