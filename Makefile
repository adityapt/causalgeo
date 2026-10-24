PYTHON ?= python3

.PHONY: install test reproduce check

install:
	$(PYTHON) -m pip install -e ".[dev,paper]"

test:
	$(PYTHON) -m pytest

reproduce:
	cd paper && $(PYTHON) reproduce.py

reproduce-fast:
	cd paper && $(PYTHON) reproduce.py --fast

check: test reproduce-fast
