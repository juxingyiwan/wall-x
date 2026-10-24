"""Documentation checks reject drift without executing an experiment."""

import os

import pytest

from scripts import check_docs


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_navigation_rejects_a_page_mentioned_only_as_code(tmp_path, monkeypatch):
    index = write(tmp_path, "README.md", "See `operator.md`.\n")
    operator = write(tmp_path, "operator.md", "# Operator\n")
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="operator.md"):
        check_docs.check_navigation([index, operator])
    index.write_text("[Operator](operator.md)\n", encoding="utf-8")
    assert check_docs.check_navigation([index, operator]) == 2


def test_links_reject_missing_anchor_but_ignore_fenced_examples(tmp_path, monkeypatch):
    index = write(tmp_path, "README.md", "[Operator](operator.md#missing)\n")
    write(tmp_path, "operator.md", "# Operator\n")
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="missing anchor"):
        check_docs.check_links([index])
    index.write_text("```text\n[Example](not-a-real-file.md)\n```\n", encoding="utf-8")
    assert check_docs.check_links([index]) == (1, 0)


def test_homing_drift_is_checked_without_importing_source(tmp_path, monkeypatch):
    page = write(
        tmp_path,
        "README.md",
        "| `front_home.MAX_ACCEL_DEG_S2` | `40.0` | acceleration |\n",
    )
    write(
        tmp_path,
        "deployment/client/src/wallx_client/hardware/front_home.py",
        "raise AssertionError('hardware source must not be imported')\nMAX_ACCEL_DEG_S2 = 40.0\n",
    )
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.check_homing_constants() == 1
    page.write_text(page.read_text().replace("40.0", "25.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from source"):
        check_docs.check_homing_constants()


def test_shell_examples_are_tokenized_without_evaluating_them():
    text = """```bash
WALLX_PADDING=original bash wallx --machine example.yml client \\
  --task "Grasp the cube" run --rtc true
```
"""
    assert list(check_docs.wallx_commands(text)) == [
        ["--machine", "example.yml", "client", "--task", "Grasp the cube", "run", "--rtc", "true"]
    ]
    with pytest.raises(ValueError, match="literal arguments"):
        list(
            check_docs.wallx_commands(
                '```bash\nbash wallx client --task "$(touch marker)" run\n```'
            )
        )


def test_cli_examples_reject_unknown_flags_and_never_dispatch(tmp_path, monkeypatch):
    from deployment import cli, local_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference or process dispatch was attempted")

    monkeypatch.setattr(cli, "verify_checkpoint", forbidden)
    monkeypatch.setattr(cli.os, "execvpe", forbidden)
    monkeypatch.setattr(local_runtime, "create_native_policy", forbidden)
    original_root = check_docs.ROOT
    for name in [
        "deployment/configs/b601.example.yml",
        "deployment/configs/client.yml",
        "deployment/machines/standalone.example.yml",
    ]:
        write(tmp_path, name, (original_root / name).read_text(encoding="utf-8"))
    pages = [
        "docs/rebot.md",
        "docs/rebot.zh-CN.md",
    ]
    for page in pages:
        write(tmp_path, page, "# Guide\n")
    guide = tmp_path / pages[0]
    guide.write_text(
        "```bash\nbash wallx --machine site.yml server\n"
        "bash wallx --machine site.yml local -- run --duration 300\n"
        "bash wallx --machine site.yml client run --rtc true\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.check_command_examples() == 3
    guide.write_text(
        "```bash\nbash wallx --machine site.yml client run --not-a-real-flag true\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parser returned"):
        check_docs.check_command_examples()


@pytest.mark.parametrize("visible_devices", ["", "4,5"])
def test_current_public_command_catalog_parses_without_changing_environment(
    monkeypatch, visible_devices
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setenv("SLURM_JOB_ID", "documentation-test")
    monkeypatch.setenv("WALLX_PADDING", "original")
    before = dict(os.environ)
    assert check_docs.check_command_examples() > 0
    assert dict(os.environ) == before


def test_bilingual_commands_cannot_drift(tmp_path, monkeypatch):
    text = '# Guide\n<a id="run"></a>\n```bash\npython command.py --steps 2\n```\n'
    write(tmp_path, "README.md", text)
    translated = write(tmp_path, "README.zh-CN.md", text.replace("Guide", "指南"))
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.check_bilingual(["README.md"]) == 1
    translated.write_text(text.replace("--steps 2", "--steps 3"), encoding="utf-8")
    with pytest.raises(ValueError, match="bilingual code blocks"):
        check_docs.check_bilingual(["README.md"])
