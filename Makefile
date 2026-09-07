PYTHON ?= python3
VERSION ?=

.PHONY: help test check format plan prepare index docker
help:
	@echo "make test       Test distribution automation (no model/native build)"
	@echo "make check      Tests, lint/format checks and basic validation"
	@echo "make format     Format Python automation and tests"
	@echo "make plan       Check upstream; optionally set VERSION=X.Y.Z"
	@echo "make prepare    Download pinned source from work/plan.json"
	@echo "make index      Generate site/ from releases.json"
	@echo "make docker     Build CPU server image; requires VERSION=X.Y.Z"

test:
	$(PYTHON) -m pytest -q

check: test
	$(PYTHON) -m ruff check .github/scripts docker tests
	$(PYTHON) -m ruff format --check .github/scripts docker tests
	$(PYTHON) -m compileall -q .github/scripts docker
	git diff --check

format:
	$(PYTHON) -m ruff format .github/scripts docker tests

plan:
	$(PYTHON) .github/scripts/check_upstream.py --version "$(VERSION)" --output work/plan.json

prepare:
	$(PYTHON) .github/scripts/prepare_source.py --plan work/plan.json --output work/prepared

index:
	$(PYTHON) .github/scripts/generate-wheel-index.py releases.json site

docker:
	test -n "$(VERSION)"
	docker build --platform linux/amd64 -f docker/simple/Dockerfile \
		--build-arg GUANACO_VERSION="$(VERSION)" -t "guanaco-py:$(VERSION)" .
