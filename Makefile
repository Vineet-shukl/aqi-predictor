.PHONY: install download sql train notebook lint test api

PY := python

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

download:
	$(PY) data/download.py

sql:
	$(PY) sql/run_analysis.py

train:
	$(PY) -m src.train

notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb

lint:
	ruff check .

test:
	pytest -q

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000
