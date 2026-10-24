"""Retired options cannot silently select another motor controller."""

import json
from dataclasses import replace

import pytest
import yaml

from wallx_client import cli
from wallx_client.config import load_config


@pytest.mark.parametrize("rtc", [False, True])
def test_legacy_false_config_preserves_current_mode(cfg, tmp_path, rtc):
    expected = replace(cfg, rtc=replace(cfg.rtc, enabled=rtc))
    data = expected.to_dict()
    data["control"]["continuous"] = False
    path = tmp_path / "legacy.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.warns(FutureWarning, match="control.continuous=false"):
        actual = load_config(path)
    assert actual == expected
    assert "continuous" not in actual.to_dict()["control"]


@pytest.mark.parametrize("value", [True, "false", 0, 1, None])
@pytest.mark.parametrize("rtc", [False, True])
def test_legacy_config_rejected_before_resources(cfg, tmp_path, monkeypatch, capsys, value, rtc):
    data = cfg.to_dict()
    data["control"]["continuous"] = value
    data["rtc"]["enabled"] = rtc
    path = tmp_path / "legacy.yml"
    path.write_text(yaml.safe_dump(data))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid legacy configuration reached a runtime resource")

    monkeypatch.setattr(cli, "Recorder", forbidden)
    monkeypatch.setattr(cli, "hardware_session", forbidden)
    assert cli.main(["--config", str(path), "run"], connection_factory=forbidden) == 1
    assert "control.continuous" in capsys.readouterr().err


@pytest.mark.parametrize("rtc", [False, True])
@pytest.mark.parametrize("before_subcommand", [False, True])
def test_legacy_false_cli_is_only_a_compatibility_input(capsys, rtc, before_subcommand):
    old = ["--continuous", "false"]
    argv = old + ["config"] if before_subcommand else ["config"] + old
    with pytest.warns(FutureWarning, match="--continuous=false"):
        assert cli.main(argv + ["--rtc", str(rtc).lower()]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "continuous" not in result["control"]
    assert result["rtc"]["enabled"] is rtc


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--continuous", "true"],
        ["--continuous=true", "run", "--rtc", "true"],
        ["--continuous=true", "run", "--continuous=false"],
        ["--continuous=false", "run", "--continuous=true"],
    ],
)
@pytest.mark.filterwarnings("ignore:--continuous=false is deprecated:FutureWarning")
def test_true_cli_cannot_be_overridden_or_reach_resources(monkeypatch, capsys, argv):
    monkeypatch.setattr(cli, "load_config", lambda _: pytest.fail("Configuration was loaded"))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert "has been removed" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"], ["config", "--help"]])
def test_help_exposes_only_current_mode_switch(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--continuous" not in help_text
    assert "--rtc" in help_text
