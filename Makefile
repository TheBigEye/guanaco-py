PYTHON ?= python3
VERSION ?=

.PHONY: help test check format plan prepare index docker
help:
	@echo "make test       Test the guanaco package (no model/native build)"
	@echo "make check      Tests, lint/format checks and basic validation"
	@echo "make format     Format the guanaco package and the tests"
	@echo "make plan       Check upstream; optionally set VERSION=X.Y.Z"
	@echo "make prepare    Download pinned source from work/plan.json"
	@echo "make index      Generate site/ from releases.json"
	@echo "make docker     Build CPU server image; requires VERSION=X.Y.Z"

test:
	$(PYTHON) -m pytest -q

check: test
	$(PYTHON) -m ruff check guanaco docker tests
	$(PYTHON) -m ruff format --check guanaco docker tests
	$(PYTHON) -m compileall -q guanaco docker
	git diff --check

format:
	$(PYTHON) -m ruff format guanaco docker tests

plan:
	$(PYTHON) -m guanaco plan --version "$(VERSION)" --output work/plan.json

prepare:
	$(PYTHON) -m pip install -q -r requirements-ci.txt
	$(PYTHON) -m guanaco prepare-source --plan work/plan.json --output work/prepared

index:
	$(PYTHON) -m guanaco build-index releases.json site

docker:
	test -n "$(VERSION)"
	docker build --platform linux/amd64 -f docker/simple/Dockerfile \
		--build-arg GUANACO_VERSION="$(VERSION)" -t "guanaco-py:$(VERSION)" .
