"""
Tests for confingy.cli module - command-line interface.
"""

import json
import sys

from confingy.cli import main

CONFIG_MODULE = """\
from dataclasses import dataclass

@dataclass
class SimpleConfig:
    name: str = "test"
    value: int = 42

config = SimpleConfig()
"""

CONFIG_FUNC_MODULE = """\
from dataclasses import dataclass

@dataclass
class SimpleConfig:
    name: str = "from_func"
    value: int = 99

def config():
    return SimpleConfig()
"""


def test_cli_serialize_basic(tmp_path, capsys, monkeypatch):
    """Test serialize command with a simple config."""
    # Create a test config file
    test_file = tmp_path / "test_config.py"
    test_file.write_text(CONFIG_MODULE)

    # Run the CLI command to serialize to stdout
    monkeypatch.setattr(sys, "argv", ["confingy", "serialize", str(test_file)])
    try:
        main()
    except SystemExit as e:
        # Typer calls sys.exit() after successful execution
        assert e.code == 0

    # Check the output
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert output["_confingy_class"] == "SimpleConfig"
    assert output["_confingy_module"] == "test_config"
    assert output["_confingy_fields"]["name"] == "test"
    assert output["_confingy_fields"]["value"] == 42


def test_cli_serialize_dotted_module_path(tmp_path, capsys, monkeypatch):
    """Test serialize with a dotted module path (pytest-style)."""
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "conf.py").write_text(CONFIG_MODULE)

    monkeypatch.setattr(sys, "argv", ["confingy", "serialize", "mypkg.conf"])
    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    output = json.loads(capsys.readouterr().out)
    assert output["_confingy_class"] == "SimpleConfig"
    assert output["_confingy_fields"]["name"] == "test"


def test_cli_serialize_dotted_module_path_with_varname(tmp_path, capsys, monkeypatch):
    """Test dotted module path with explicit ::variable_name."""
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "conf.py").write_text(
        CONFIG_MODULE + "\nmy_var = SimpleConfig(name='custom', value=7)\n"
    )

    monkeypatch.setattr(sys, "argv", ["confingy", "serialize", "mypkg.conf::my_var"])
    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    output = json.loads(capsys.readouterr().out)
    assert output["_confingy_fields"]["name"] == "custom"
    assert output["_confingy_fields"]["value"] == 7


def test_cli_serialize_dotted_module_init(tmp_path, capsys, monkeypatch):
    """Test dotted module path resolving to __init__.py."""
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "mypkg" / "sub"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(CONFIG_MODULE)

    monkeypatch.setattr(sys, "argv", ["confingy", "serialize", "mypkg.sub"])
    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    output = json.loads(capsys.readouterr().out)
    assert output["_confingy_class"] == "SimpleConfig"
    assert output["_confingy_module"] == "mypkg.sub"
    assert output["_confingy_fields"]["value"] == 42


def test_cli_serialize_auto_call_function(tmp_path, capsys, monkeypatch):
    """Test that functions are auto-called and return value is serialized."""
    test_file = tmp_path / "func_config.py"
    test_file.write_text(CONFIG_FUNC_MODULE)

    monkeypatch.setattr(sys, "argv", ["confingy", "serialize", str(test_file)])
    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    output = json.loads(capsys.readouterr().out)
    assert output["_confingy_class"] == "SimpleConfig"
    assert output["_confingy_fields"]["name"] == "from_func"
    assert output["_confingy_fields"]["value"] == 99


def test_cli_serialize_dotted_path_with_auto_call(tmp_path, capsys, monkeypatch):
    """Test dotted module path combined with function auto-call."""
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "conf.py").write_text(CONFIG_FUNC_MODULE)

    monkeypatch.setattr(sys, "argv", ["confingy", "serialize", "mypkg.conf"])
    try:
        main()
    except SystemExit as e:
        assert e.code == 0

    output = json.loads(capsys.readouterr().out)
    assert output["_confingy_class"] == "SimpleConfig"
    assert output["_confingy_fields"]["name"] == "from_func"
    assert output["_confingy_fields"]["value"] == 99
