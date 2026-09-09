"""Build system for the ``guanaco-py`` wheel distribution.

This package replaces what used to be fourteen loose scripts under
``.github/scripts/``. Every module here has a single, clearly named job:

===========================  ==================================================
Module                       Responsibility
===========================  ==================================================
:mod:`guanaco.settings`      Load and validate every configurable value.
:mod:`guanaco.models`        Value objects and the JSON documents we exchange.
:mod:`guanaco.transfer`      Bounded downloads and safe archive extraction.
:mod:`guanaco.github_api`    Read-mostly GitHub REST client.
:mod:`guanaco.source`        Build the immutable, checksummed source snapshot.
:mod:`guanaco.toolchain`     Compiler flags and build matrix selection.
:mod:`guanaco.wheels`        Wheel integrity, contents and receipts.
:mod:`guanaco.releases`      Upstream discovery, planning and publication.
:mod:`guanaco.catalog`       PEP 503 wheel index rendered to GitHub Pages.
:mod:`guanaco.reports`       Downloadable diagnostics for test builds.
:mod:`guanaco.cli`           The one command line entry point.
===========================  ==================================================

Nothing in this package hardcodes a repository name, a package name or a
version number: every value is read from ``.github/build-matrix.json`` (or an
environment variable) through :class:`guanaco.settings.Settings`, so renaming
the repository or the distribution is a configuration change, not a code
change.

The package is imported by the Docker helpers as well, which copy it next to
``docker/fetch_release.py`` and set ``GUANACO_ROOT``.
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
