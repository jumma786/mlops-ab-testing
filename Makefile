.PHONY: install train test run clean

install:
	pip install -r requirements.txt

train:
	python src/models/train.py --data-path data/bank-additional-full.csv

train-synthetic:
	python src/models/train.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run:
	python -m uvicorn src.router.app:app --host 127.0.0.1 --port 8002 --reload

clean:
	rm -rf artifacts/ __pycache__
	find . -name "*.pyc" -delete
