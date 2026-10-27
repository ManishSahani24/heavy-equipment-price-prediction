# Data

The competition data is **not** included in this repository (it is subject to the Kaggle competition
rules of a private course competition).

Place the files here:

```
data/raw/
├── train.csv              # 138,701 rows × 50 columns
├── test.csv               # 15,000 rows × 49 columns
├── sample_submission.csv
└── metaData.csv
```

With competition access and a Kaggle API token (`~/.kaggle/kaggle.json`):

```bash
bash scripts/download_data.sh
```

No access? Generate schema-identical synthetic data to run the pipeline end-to-end:

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python scripts/train.py --data-dir data/synthetic --quick
```

Column descriptions: [`docs/data_dictionary.md`](../docs/data_dictionary.md).
