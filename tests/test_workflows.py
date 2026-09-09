"""Static contracts for the workflow files themselves.

CI lints the workflows with actionlint, which catches a reference to an
output a job never declared. This module checks the same contract locally
with PyYAML, so the mistake is caught before it reaches a runner.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))

NEEDS_OUTPUT = re.compile(r"needs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)")
NEEDS_JOB = re.compile(r"needs\.([A-Za-z0-9_-]+)")


def _walk(node):
    """Yield every string inside a parsed workflow document."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                yield key
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)
    elif isinstance(node, str):
        yield node


def _jobs(path: Path) -> dict:
    """Return the ``jobs`` mapping of one workflow file."""
    with path.open(encoding="utf-8") as stream:
        return (yaml.safe_load(stream) or {}).get("jobs") or {}


def _identifiers(name: str) -> set[str]:
    """Return the spellings GitHub accepts for one job identifier."""
    return {str(name), str(name).replace("-", "_")}


def test_there_are_workflows_to_check():
    assert WORKFLOWS, "no workflow files found"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_consumed_output_is_declared(path: Path):
    """A job must declare an output before another job can read it."""
    jobs = _jobs(path)
    problems = []
    for consumer, definition in jobs.items():
        for text in _walk(definition):
            for job, output in NEEDS_OUTPUT.findall(text):
                if job not in jobs:
                    continue
                declared = {str(key) for key in (jobs[job].get("outputs") or {})}
                if output not in declared:
                    problems.append(f"{consumer} reads {job}.{output}")
    assert not problems, f"{path.name}: undeclared output(s): {sorted(set(problems))}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_job_it_needs_exists(path: Path):
    """``needs:`` must name a job the same workflow defines."""
    jobs = _jobs(path)
    known: set[str] = set()
    for name in jobs:
        known |= _identifiers(name)
    problems = []
    for consumer, definition in jobs.items():
        needs = definition.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        for name in needs:
            if str(name) not in known:
                problems.append(f"{consumer} needs unknown job {name}")
        for text in _walk(definition):
            for name in NEEDS_JOB.findall(text):
                if name not in known:
                    problems.append(f"{consumer} references needs.{name}")
    assert not problems, f"{path.name}: {sorted(set(problems))}"
