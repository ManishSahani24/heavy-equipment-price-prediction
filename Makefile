.PHONY: install data eda train demo test clean

install:
	pip install -r requirements.txt

data:
	bash scripts/download_data.sh

eda:
	python scripts/eda.py

train:
	python scripts/train.py

# end-to-end run on synthetic data (~20 s, no Kaggle access needed)
demo:
	python scripts/make_synthetic_data.py --out data/synthetic
	python scripts/train.py --data-dir data/synthetic --output-dir outputs/demo --quick

test:
	python -m pytest -q

clean:
	rm -rf outputs/demo data/synthetic catboost_info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
