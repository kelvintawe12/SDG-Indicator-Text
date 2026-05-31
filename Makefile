.PHONY: install test smoke run-all clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

smoke:
	python scripts/smoke_test.py

run-all:
	python scripts/run_all_experiments.py
	python scripts/make_diagnostics.py
	python scripts/predict_test_set.py

clean:
	rm -rf artifacts/models/* artifacts/predictions/* data/interim/*
