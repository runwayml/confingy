"""Tests for confingy.utils.imports helper functions."""

import pytest

from confingy.utils.imports import (
    _is_file_path,
    _resolve_module_path,
    derive_module_name,
)


class TestIsFilePath:
    def test_path_with_slash(self):
        assert _is_file_path("path/to/module.py") is True

    def test_path_ending_with_py(self):
        assert _is_file_path("module.py") is True

    def test_dotted_module_path(self):
        assert _is_file_path("path.to.module") is False

    def test_single_name(self):
        assert _is_file_path("module") is False

    def test_slash_without_py(self):
        assert _is_file_path("path/to/module") is True


class TestResolveModulePath:
    def test_resolves_py_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "foo").mkdir()
        (tmp_path / "foo" / "bar.py").write_text("x = 1\n")

        result = _resolve_module_path("foo.bar")
        assert result.name == "bar.py"

    def test_resolves_init_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "foo" / "bar").mkdir(parents=True)
        (tmp_path / "foo" / "bar" / "__init__.py").write_text("x = 1\n")

        result = _resolve_module_path("foo.bar")
        assert result.name == "__init__.py"

    def test_prefers_py_over_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "foo").mkdir()
        (tmp_path / "foo" / "bar.py").write_text("x = 1\n")
        (tmp_path / "foo" / "bar").mkdir()
        (tmp_path / "foo" / "bar" / "__init__.py").write_text("x = 2\n")

        result = _resolve_module_path("foo.bar")
        assert result.name == "bar.py"

    def test_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Cannot resolve module path"):
            _resolve_module_path("nonexistent.module")


class TestDeriveModuleName:
    def test_init_py_in_subpackage(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pkg = tmp_path / "mypkg" / "sub"
        pkg.mkdir(parents=True)
        init = pkg / "__init__.py"
        init.write_text("")

        assert derive_module_name(init) == "mypkg.sub"

    def test_root_init_py(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        init = tmp_path / "__init__.py"
        init.write_text("")

        assert derive_module_name(init) == "__init__"

    def test_regular_py_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pkg").mkdir()
        py_file = tmp_path / "pkg" / "mod.py"
        py_file.write_text("")

        assert derive_module_name(py_file) == "pkg.mod"
