"""Wheel validation: filename, metadata, RECORD, runtime and native libraries."""

from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import (
    SHA_B,
    make_plan,
    manifest_for,
    native_header,
    write_wheel,
    zip_contents,
)
from source_helpers import raw_zip_member

from guanaco.models import Channel, Platform
from guanaco.wheels import WheelError, WheelValidator


@pytest.fixture
def prepared(settings):
    """A plan plus the manifest and runtime files it describes."""
    plan = make_plan(settings)
    manifest, runtime = manifest_for(plan)
    return plan, manifest, runtime


def build(validator, tmp_path, prepared, **options):
    """Write one wheel and validate it, returning its identity."""
    plan, manifest, runtime = prepared
    wheel = write_wheel(tmp_path / "wheel", manifest, runtime, **options)
    return validator.verify(
        wheel, manifest, Channel(options["channel"]), Platform(options["platform"])
    )


class TestFilenames:
    def test_a_valid_wheel_passes(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        assert WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_wrong_platform_tag_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        with pytest.raises(WheelError, match="Unexpected wheel filename"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.WINDOWS)

    def test_an_unrepaired_cpu_wheel_is_rejected_by_default(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313", raw=True)
        with pytest.raises(WheelError):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_an_unrepaired_cpu_wheel_can_be_allowed_locally(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313", raw=True)
        assert WheelValidator(settings).verify(
            wheel, manifest, Channel("cpu"), Platform.LINUX, allow_unrepaired=True
        )

    def test_a_python_version_outside_the_matrix_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp37")
        with pytest.raises(WheelError):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_cuda_channels_use_an_unrepaired_tag(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cu124", "linux", "cp313")
        assert (
            WheelValidator(settings)
            .verify(wheel, manifest, Channel("cu124"), Platform.LINUX)
            .platform
            == "linux_x86_64"
        )


class TestContents:
    def test_an_extra_python_file_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313", extra=True)
        with pytest.raises(WheelError, match="missing or additional Python binding"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_an_altered_binding_file_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313", altered=True)
        with pytest.raises(WheelError, match="Upstream binding changed"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_renamed_metadata_directory_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = {
            name.replace("guanaco_py-0.3.49.dist-info", "other-0.3.49.dist-info"): data
            for name, data in _members(wheel).items()
        }
        zip_contents(wheel, contents)
        with pytest.raises(WheelError, match="renamed without rebuilding"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_wrong_distribution_name_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(
            tmp_path, manifest, runtime, "cpu", "linux", "cp313", metadata_name="llama-cpp-python"
        )
        with pytest.raises(WheelError, match="distribution name/version mismatch"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_an_upstream_dependency_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = _members(wheel)
        info = "guanaco_py-0.3.49.dist-info/METADATA"
        contents[info] = contents[info] + b"Requires-Dist: llama_cpp_python==0.3.49\n"
        zip_contents(wheel, contents)
        with pytest.raises(WheelError, match="upstream distribution itself"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_tampered_file_breaks_the_record(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = _members(wheel)
        contents["llama_cpp/py.typed"] = b"tampered"
        zip_contents(wheel, contents, record=False)
        with pytest.raises(WheelError, match="RECORD integrity mismatch"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_missing_record_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = _members(wheel)
        del contents["guanaco_py-0.3.49.dist-info/RECORD"]
        zip_contents(wheel, contents, record=False)
        with pytest.raises(WheelError, match="missing RECORD"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_missing_native_library_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313", native=False)
        with pytest.raises(WheelError, match="no native runtime libraries"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_broken_elf_header_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = _members(wheel)
        contents["llama_cpp/lib/libllama.so"] = b"not an elf"
        zip_contents(wheel, contents)
        with pytest.raises(WheelError, match="Invalid x86-64 native library header"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_windows_needs_a_pe_header(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "windows", "cp313")
        contents = _members(wheel)
        contents["llama_cpp/lib/llama.dll"] = native_header("linux")
        zip_contents(wheel, contents)
        with pytest.raises(WheelError, match="Invalid x86-64 native library header"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.WINDOWS)

    def test_a_missing_license_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = _members(wheel)
        del contents["guanaco_py-0.3.49.dist-info/licenses/LICENSE.md"]
        zip_contents(wheel, contents)
        with pytest.raises(WheelError, match="missing its license notice"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_path_traversal_member_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = _members(wheel)
        zip_contents(wheel, contents)
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr(raw_zip_member("../../../escape.py"), b"nope")
        with pytest.raises(WheelError):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_data_directory_that_overrides_the_runtime_is_rejected(
        self, tmp_path, settings, prepared
    ):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        contents = _members(wheel)
        contents["guanaco_py-0.3.49.data/platlib/llama_cpp/__init__.py"] = b"override"
        zip_contents(wheel, contents)
        with pytest.raises(WheelError, match="could override the verified runtime"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)

    def test_a_corrupt_archive_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux", "cp313")
        wheel.write_bytes(b"not a zip")
        with pytest.raises(WheelError, match="Invalid wheel archive"):
            WheelValidator(settings).verify(wheel, manifest, Channel("cpu"), Platform.LINUX)


class TestDirectory:
    def test_a_complete_matrix_passes(self, tmp_path, settings):
        plan = replace(make_plan(settings), matrix=narrow_matrix(settings, "3.12", "3.13"))
        manifest, runtime = manifest_for(plan)
        directory = tmp_path / "wheels"
        for python in ("cp312", "cp313"):
            write_wheel(directory, manifest, runtime, "cpu", "linux", python)
        wheels = WheelValidator(settings).verify_directory(
            directory, manifest, Channel("cpu"), Platform.LINUX
        )
        assert len(wheels) == 2

    def test_an_incomplete_matrix_is_rejected(self, tmp_path, settings, prepared):
        _, manifest, runtime = prepared
        directory = tmp_path / "wheels"
        write_wheel(directory, manifest, runtime, "cpu", "linux", "cp313")
        with pytest.raises(WheelError, match="Expected 6 wheels"):
            WheelValidator(settings).verify_directory(
                directory, manifest, Channel("cpu"), Platform.LINUX
            )

    def test_a_receipt_describes_the_wheels(self, tmp_path, settings):
        plan = replace(make_plan(settings), matrix=narrow_matrix(settings, "3.13"))
        manifest, runtime = manifest_for(plan)
        directory = tmp_path / "wheels"
        write_wheel(directory, manifest, runtime, "cpu", "linux", "cp313")
        validator = WheelValidator(settings)
        wheels = validator.verify_directory(directory, manifest, Channel("cpu"), Platform.LINUX)
        receipt = validator.receipt(wheels, manifest, Channel("cpu"), Platform.LINUX)
        assert receipt.version == manifest.version
        assert receipt.recipe_commit == SHA_B
        assert receipt.wheels[0].name.endswith(".whl")


def narrow_matrix(settings, *python_versions):
    """Return the configured matrix reduced to a few Python versions."""
    return replace(settings.matrix, python_versions=tuple(python_versions))


def _members(wheel: Path) -> dict:
    """Read a wheel's members into memory."""
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist()}
